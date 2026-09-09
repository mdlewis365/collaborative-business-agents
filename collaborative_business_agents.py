import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from autogen_agentchat.agents import AssistantAgent, BaseChatAgent, UserProxyAgent
from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.conditions import FunctionalTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from dotenv import load_dotenv


APPROVAL_COST_THRESHOLD = 1500.00
SPECIAL_LEAVE_TYPES = {"extended leave", "unpaid leave", "policy exception"}
HR_COMPLETE_SENTENCE = "HR complete, please continue."
PAUSE_MESSAGE = "Manager Approval Needed - Task Paused"

LABEL_PATTERN = re.compile(
    r"^\[From:\s*(?P<sender>[^\]]+)\]"
    r"\[Task:\s*(?P<task>[^\]]+)\]"
    r"\[Status:\s*(?P<status>[^\]]+)\]"
    r"\[Next:\s*(?P<next_agent>[^\]]+)\]"
)
HR_POLICY_PATTERN = re.compile(
    r"^HR policy check: (PASSED|FAILED)\s*$", re.MULTILINE
)


class WorkflowError(ValueError):
    """The workflow cannot proceed with the supplied data or message."""


def parse_labels(content: str) -> dict[str, str]:
    match = LABEL_PATTERN.match(content)
    if match is None:
        raise WorkflowError(
            "The message must start with "
            "[From: ...][Task: ...][Status: ...][Next: ...]."
        )
    return {key: value.strip() for key, value in match.groupdict().items()}


def create_approval_state(
    leave_type: str, requested_days: int, daily_cost: float
) -> dict[str, Any]:
    """Validate the request and compute the existing manager-review rule."""
    if not isinstance(leave_type, str) or not leave_type.strip():
        raise ValueError("leave_type must be a nonempty string.")
    if type(requested_days) is not int or requested_days <= 0:
        raise ValueError("requested_days must be a positive whole number.")
    if (
        isinstance(daily_cost, bool)
        or not isinstance(daily_cost, (int, float))
        or not math.isfinite(daily_cost)
        or daily_cost < 0
    ):
        raise ValueError("daily_cost must be a finite, nonnegative number.")

    leave_type = leave_type.strip()
    estimated_cost = round(requested_days * daily_cost, 2)
    if not math.isfinite(estimated_cost):
        raise ValueError("The estimated cost is too large.")
    reasons = []
    if estimated_cost >= APPROVAL_COST_THRESHOLD:
        reasons.append(
            f"Estimated cost ${estimated_cost:,.2f} meets or exceeds "
            f"the ${APPROVAL_COST_THRESHOLD:,.2f} approval threshold."
        )
    if leave_type.lower() in SPECIAL_LEAVE_TYPES:
        reasons.append(f"{leave_type} requires special manager approval.")

    return {
        "leave_type": leave_type,
        "requested_days": requested_days,
        "daily_cost": daily_cost,
        "required": bool(reasons),
        "reason": " ".join(reasons),
        "estimated_cost": estimated_cost,
        "decision": "Pending" if reasons else "Not Required",
    }


@dataclass
class WorkflowState:

    approval_state: dict[str, Any] = field(default_factory=dict)
    hr_policy_passed: bool | None = None
    outcome: str = "Pending"
    stop_reason: str = ""

    def create_and_save_approval_state(
        self, leave_type: str, requested_days: int, daily_cost: float
    ) -> str:
        """Save supplied leave details and return the manager-review requirement.

        Saving a request does not approve it. Do not guess missing arguments.
        """
        new_state = create_approval_state(leave_type, requested_days, daily_cost)
        if self.approval_state:
            fields = ("leave_type", "requested_days", "daily_cost")
            if any(self.approval_state[key] != new_state[key] for key in fields):
                raise ValueError("A different request is already saved in this workflow.")
        else:
            self.approval_state.update(new_state)
        return json.dumps({"saved": True, "approval_state": self.approval_state})


HR_SYSTEM_MESSAGE = """
You are HR_Assistant in a sequential leave-processing workflow.
The employee's submitted request is already confirmed. Do not ask the employee
to confirm it again.

1. Read the employee's request and supplied workflow data. Identify leave_type,
   requested_days, daily_cost, requested hours, available PTO, and leave policy.
   Do not invent values. Convert hours to days only using supplied working
   hours per day. The current tool accepts positive whole days only; if a
   request cannot be represented exactly, report it as blocked.

2. Check requested hours against available PTO and apply the supplied policy.
   Determine whether the HR policy check PASSED or FAILED and explain why.

3. If the HR check passes and the required data is available, make an actual
   call to create_and_save_approval_state with leave_type, requested_days,
   and the supplied daily_cost. Merely mentioning the function is not a call.
   Do not repeat a successful call for this request. Read the tool result.
   Report the state as saved only if the tool confirms success.

4. Forward the raw financial information to Finance_Assistant. Do not calculate
   total cost or budget impact yourself. The Python tool computes the canonical
   manager-review threshold; Finance prepares the financial report.

FINAL TEXT CONTRACT (applies after tool execution, not to tool-call events):
The first character must be "[". No greeting or Markdown before the header.

After a passed HR check and a successful tool call, begin exactly with:
[From: HR][Task: Approve Leave][Status: Complete][Next: Finance]

Include this exact line:
HR policy check: PASSED

Include the policy reason, leave type, requested days and hours, available PTO,
supplied daily cost, department budget, and other raw data Finance needs.
End with this exact sentence:
HR complete, please continue.

If the supplied policy check fails, begin exactly with:
[From: HR][Task: Approve Leave][Status: Rejected][Next: End]
Include this exact line:
HR policy check: FAILED
Explain the policy failure. Do not send a failed HR request to Finance.

If data is missing, ambiguous, or a tool call fails, begin exactly with:
[From: HR][Task: Approve Leave][Status: Blocked][Next: None]
Identify the missing information or error. Ask only for missing facts, not
reconfirmation. Do not claim completion or use the completion sentence.
""".strip()

FINANCE_SYSTEM_MESSAGE = f"""
You are Finance_Assistant in a sequential leave-processing workflow.
Use the supplied request and HR handoff to report requested leave days,
the supplied daily employee cost, total cost, and budget remaining afterward.
Do not invent missing financial data. The saved tool result contains the
canonical estimated cost and manager-review requirement.

Manager review is required when the estimated cost is at least
${APPROVAL_COST_THRESHOLD:,.2f}, or for extended leave, unpaid leave, or a
policy exception. Do not approve or reject requests requiring human review.

If financial data is missing or contradictory, begin exactly with:
[From: Finance][Task: Calculate Leave Impact][Status: Blocked][Next: None]
Explain what is needed; do not claim completion.

Otherwise begin with the appropriate header:
[From: Finance][Task: Calculate Leave Impact][Status: Approval Required][Next: Manager]
or:
[From: Finance][Task: Calculate Leave Impact][Status: Complete][Next: End]

Then provide your financial report. The application applies the final routing
header using the saved Python state. Do not add a greeting or Markdown before
the header. Do not ask the employee to confirm the request again.
""".strip()


class FinanceProtocolAgent(BaseChatAgent):
    
    def __init__(self, model_client: ChatCompletionClient, state: WorkflowState):
        super().__init__(
            name="Finance_Assistant",
            description="Reports leave costs and routes required human approvals.",
        )
        self._state = state
        self._assistant = AssistantAgent(
            name=self.name,
            model_client=model_client,
            system_message=FINANCE_SYSTEM_MESSAGE,
        )

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return [TextMessage]

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        state = self._state.approval_state
        if not state or self._state.hr_policy_passed is not True:
            raise WorkflowError("Finance requires a saved request and a passed HR check.")

        response = await self._assistant.on_messages(messages, cancellation_token)
        message = response.chat_message
        if not isinstance(message, TextMessage):
            raise WorkflowError("Finance must return a text report.")

        match = LABEL_PATTERN.match(message.content)
        if match and match.group("status").strip() == "Blocked":
            return response

        body = LABEL_PATTERN.sub("", message.content, count=1).strip()
        if not body:
            raise WorkflowError("Finance returned an empty report.")
        if state["required"]:
            header = (
                "[From: Finance][Task: Calculate Leave Impact]"
                "[Status: Approval Required][Next: Manager]"
            )
            if PAUSE_MESSAGE not in body:
                body = f"{body}\n\n{PAUSE_MESSAGE}"
        else:
            header = (
                "[From: Finance][Task: Calculate Leave Impact]"
                "[Status: Complete][Next: End]"
            )
            body = body.replace(PAUSE_MESSAGE, "").strip()

        return Response(
            chat_message=message.model_copy(update={"content": f"{header}\n{body}"}),
            inner_messages=response.inner_messages,
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        await self._assistant.on_reset(cancellation_token)


EXAMPLE_REQUEST = """
[From: Employee][Task: Submit Leave][Status: Ready][Next: HR]
I am requesting 40 hours of paid leave from September 14 through
September 18, 2026.

Leave type: paid leave
Requested paid-leave days: 5
Working hours per day: 8
Available PTO balance: 80 hours
Daily employee cost: $320
Department leave-impact budget remaining: $10,000

Applicable policy: Requests of 12 days or fewer may be approved when
the employee has enough available PTO.
""".strip()


class LeaveWorkflow:
    """One request, its three participants, and its deterministic routing."""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        *,
        read_manager_input: Callable[[str], str] = input,
    ):
        self.state = WorkflowState()
        self._read_manager_input = read_manager_input
        self._started = False
        self.hr_assistant = AssistantAgent(
            name="HR_Assistant",
            description="Checks leave policy and saves the request using its tool.",
            model_client=model_client,
            system_message=HR_SYSTEM_MESSAGE,
            tools=[self.state.create_and_save_approval_state],
            reflect_on_tool_use=True,
            max_tool_iterations=1,
        )
        self.finance_assistant = FinanceProtocolAgent(model_client, self.state)
        self.manager_approver = UserProxyAgent(
            name="Manager_Approver",
            description="A human manager who approves or rejects flagged requests.",
            input_func=self._manager_input,
        )

        self.team = SelectorGroupChat(
            participants=[
                self.hr_assistant, self.finance_assistant, self.manager_approver
            ],
            model_client=model_client,
            selector_func=self.select_next_speaker,
            termination_condition=FunctionalTermination(self.should_stop),
            max_turns=6,
            allow_repeated_speaker=False,
        )

    def _manager_input(self, prompt: str) -> str:
        state = self.state.approval_state
        if (
            not state
            or not state["required"]
            or self.state.hr_policy_passed is not True
        ):
            raise WorkflowError("No eligible request requires manager approval.")
        print(f"\n{PAUSE_MESSAGE}")
        print(f"Reason: {state['reason']}")
        print(f"Estimated financial impact: ${state['estimated_cost']:,.2f}")
        while True:
            answer = self._read_manager_input(
                "Manager decision—type APPROVE or REJECT: "
            ).strip().lower()
            if answer in {"approve", "approved", "a"}:
                decision = "Approved"
            elif answer in {"reject", "rejected", "r"}:
                decision = "Rejected"
            else:
                print("Invalid response. Please type APPROVE or REJECT.")
                continue
            state["decision"] = decision
            return (
                "[From: Manager][Task: Approve Leave Expense]"
                f"[Status: {decision}][Next: End]\n"
                f"The manager {decision.lower()} the request and its financial impact."
            )

    def _route(self, messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
        
        last = next(
            (item for item in reversed(messages) if isinstance(item, BaseChatMessage)),
            None,
        )
        if not isinstance(last, TextMessage):
            raise WorkflowError("Expected a final text message for workflow routing.")
        labels = parse_labels(last.content)

        expected_senders = {
            "Employee": "Employee",
            "HR_Assistant": "HR",
            "Finance_Assistant": "Finance",
            "Manager_Approver": "Manager",
        }
        if last.source not in expected_senders or labels["sender"] != expected_senders[last.source]:
            raise WorkflowError("The message's From label does not match its actual source.")

        if last.source == "Employee":
            if labels["status"] != "Ready" or labels["next_agent"] != "HR":
                raise WorkflowError("The employee request must be Ready and assigned to HR.")
            return "HR_Assistant"

        if last.source in {"HR_Assistant", "Finance_Assistant"} and labels["status"] == "Blocked":
            if labels["next_agent"] != "None":
                raise WorkflowError("A blocked request must use [Next: None].")
            self.state.outcome = "Blocked"
            self.state.stop_reason = f"{last.source} needs missing information or a correction."
            return None

        if last.source == "HR_Assistant":
            policy_lines = HR_POLICY_PATTERN.findall(last.content)
            if len(policy_lines) != 1:
                raise WorkflowError("HR must supply exactly one HR policy check: PASSED/FAILED line.")
            self.state.hr_policy_passed = policy_lines[0] == "PASSED"
            if not self.state.hr_policy_passed:
                self.state.outcome = "Rejected by HR"
                self.state.stop_reason = "The HR policy check failed; Finance was not run."
                return None
            if (
                labels["status"] != "Complete"
                or labels["next_agent"] != "Finance"
                or not last.content.rstrip().endswith(HR_COMPLETE_SENTENCE)
            ):
                raise WorkflowError("Invalid HR-to-Finance handoff.")
            if not self.state.approval_state:
                raise WorkflowError("HR did not successfully execute create_and_save_approval_state.")
            return "Finance_Assistant"

        if last.source == "Finance_Assistant":
            state = self.state.approval_state
            if not state or self.state.hr_policy_passed is not True:
                raise WorkflowError("Finance requires a saved request and a passed HR check.")
            if state["required"]:
                if labels["status"] != "Approval Required" or labels["next_agent"] != "Manager":
                    raise WorkflowError("Finance must route this request to the manager.")
                return "Manager_Approver"
            if labels["status"] != "Complete" or labels["next_agent"] != "End":
                raise WorkflowError("Invalid Finance completion message.")
            self.state.outcome = "Completed"
            self.state.stop_reason = "HR and Finance completed; manager approval was not required."
            return None

        if last.source == "Manager_Approver":
            state = self.state.approval_state
            if (
                not state
                or not state["required"]
                or labels["status"] not in {"Approved", "Rejected"}
                or labels["next_agent"] != "End"
                or state["decision"] != labels["status"]
            ):
                raise WorkflowError("The manager message does not match the recorded human decision.")
            self.state.outcome = state["decision"]
            self.state.stop_reason = f"Final manager decision: {state['decision']}."
            return None

        raise WorkflowError("No valid next step exists for this message.")

    def should_stop(self, messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> bool:
        try:
            return self._route(messages) is None
        except WorkflowError as error:
            self.state.outcome = "Blocked"
            self.state.stop_reason = str(error)
            return True

    def select_next_speaker(self, messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str:
        next_speaker = self._route(messages)
        if next_speaker is None:
            raise WorkflowError("The termination condition should have stopped this workflow.")
        return next_speaker

    async def run(self, request_text: str = EXAMPLE_REQUEST, *, show_console: bool = True) -> TaskResult:
        if self._started:
            raise RuntimeError("Create a new LeaveWorkflow for each new request.")
        self._started = True
        task = TextMessage(source="Employee", content=request_text.strip())
        if show_console:
            result = await Console(self.team.run_stream(task=task))
        else:
            result = await self.team.run(task=task)
        if self.state.outcome == "Pending":
            self.state.outcome = "Blocked"
            self.state.stop_reason = result.stop_reason or "The team stopped before completion."
        return result


def create_model_client() -> AzureOpenAIChatCompletionClient:
    """Replace llm_config with the newer Azure model-client configuration."""
    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION")
    required = {
        "AZURE_OPENAI_API_KEY": api_key,
        "AZURE_OPENAI_ENDPOINT": endpoint,
        "AZURE_OPENAI_DEPLOYMENT": deployment,
        "API_VERSION (or AZURE_OPENAI_API_VERSION)": api_version,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError("Missing configuration in the environment or adjacent .env: " + ", ".join(missing))

    model_name = os.getenv("AZURE_OPENAI_MODEL") or deployment
    try:
        return AzureOpenAIChatCompletionClient(
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_key=api_key,
            api_version=api_version,
            model=model_name,
        )
    except ValueError as error:
        if "model_info is required" in str(error):
            raise RuntimeError(
                "Set AZURE_OPENAI_MODEL to the underlying model name, "
                "rather than a custom Azure deployment alias."
            ) from error
        raise


async def main() -> None:
    model_client = create_model_client()
    try:
        workflow = LeaveWorkflow(model_client)
        await workflow.run(EXAMPLE_REQUEST)
        print(f"\nWorkflow outcome: {workflow.state.outcome}")
        print(workflow.state.stop_reason)
        print("Saved approval state:")
        print(json.dumps(workflow.state.approval_state, indent=2))
    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
