import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, GroupChatManager, GroupChat
import re

load_dotenv()
OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

llm_config = {
    "config_list": [{
            "api_type": "azure",
        	"api_version": API_VERSION,
        	"api_key": OPENAI_KEY,
        	"base_url": OPENAI_ENDPOINT,
            "model": AZURE_OPENAI_DEPLOYMENT
}]
}

LABEL_PATTERN = re.compile(
    r"^\[From:\s*(?P<sender>[^\]]+)\]"
    r"\[Task:\s*(?P<task>[^\]]+)\]"
    r"\[Status:\s*(?P<status>[^\]]+)\]"
    r"\[Next:\s*(?P<next_agent>[^\]]+)\]"
)

def parse_labels(content):

    match = LABEL_PATTERN.match(str(content).strip())

    if not match:
        raise ValueError(
            "Message must begin with "
            "[From: ...][Task: ...][Status: ...][Next: ...]"
        )

    return match.groupdict()

APPROVAL_COST_THRESHOLD = 1500.00

SPECIAL_LEAVE_TYPES = {
    "extended leave",
    "unpaid leave",
    "policy exception",
}

def create_approval_state(
    leave_type,
    requested_days,
    daily_cost,
):
    estimated_cost = requested_days * daily_cost

    reasons = []

    if estimated_cost >= APPROVAL_COST_THRESHOLD:
        reasons.append(
            f"Estimated cost ${estimated_cost:,.2f} meets or exceeds "
            f"the ${APPROVAL_COST_THRESHOLD:,.2f} approval threshold."
        )

    if leave_type.lower() in SPECIAL_LEAVE_TYPES:
        reasons.append(f"{leave_type} requires special manager approval.")

    return {
        "required": bool(reasons),
        "reason": " ".join(reasons),
        "estimated_cost": estimated_cost,
        "decision": "Pending" if reasons else "Not Required",
    }

approval_state = create_approval_state(
    leave_type="PTO",
    requested_days=5,
    daily_cost=320.00,
)

def alert_manager(approval_state):
    """Display the approval alert in the shared console."""

    print("\n" + "=" * 70)
    print("Manager Approval Needed - Task Paused")
    print(f"Reason: {approval_state['reason']}")
    print(
        f"Estimated financial impact: "
        f"${approval_state['estimated_cost']:,.2f}"
    )
    print("=" * 70)

class ManagerApprovalAgent(UserProxyAgent):
    """Human-in-the-loop approval agent."""

    def __init__(self, approval_state):
        self.approval_state = approval_state

        super().__init__(
            name="Manager_Approver",
            description=(
                "A human manager who approves or rejects requests that "
                "exceed an expense threshold or require special approval."
            ),
            human_input_mode="ALWAYS",
            code_execution_config=False,
            llm_config=False,
        )

    def get_human_input(self, prompt, iostream=None, **kwargs):

        _ = (prompt, kwargs)
        alert_manager(self.approval_state)

        read_input = iostream.input if iostream is not None else input

        while True:
            decision = read_input(
                "Manager decision—type APPROVE or REJECT: "
            ).strip().lower()

            if decision in {"approve", "approved", "a"}:
                self.approval_state["decision"] = "Approved"

                return (
                    "[From: Manager]"
                    "[Task: Approve Leave Expense]"
                    "[Status: Approved]"
                    "[Next: End]\n"
                    "The manager approved the leave request and its "
                    "financial impact."
                )

            if decision in {"reject", "rejected", "r"}:
                self.approval_state["decision"] = "Rejected"

                return (
                    "[From: Manager]"
                    "[Task: Approve Leave Expense]"
                    "[Status: Rejected]"
                    "[Next: End]\n"
                    "The manager rejected the leave request based on its "
                    "financial impact."
                )

            print("Invalid response. Please type APPROVE or REJECT.")

manager_approver = ManagerApprovalAgent(approval_state)

hr_assistant = AssistantAgent(
    name="HR_Assistant",
    description=(
        "HR Assistant handles employee questions, HR policy information, "
        "policy-management requests, and employee leave requests."
    ),
    system_message="""
You are HR_Assistant in a sequential leave-processing workflow.

The employee's submitted request is already confirmed. Do not ask the
employee to confirm it again.

Your responsibilities:
1. Check the requested hours against the available PTO balance.
2. Apply the supplied leave policy.
3. Record whether the HR policy check passed.
4. Forward the raw financial information to Finance_Assistant.
5. Do not calculate cost or budget impact; Finance performs that work.

STRICT RESPONSE CONTRACT:
- The first character of your response must be "[".
- Do not place a greeting, introduction, or Markdown before the header.
- Begin with this exact line:

[From: HR][Task: Approve Leave][Status: Complete][Next: Finance]

- Include the HR decision and the raw information Finance needs.
- End with this exact sentence:

HR complete, please continue.
""",
    llm_config=llm_config,
)

finance_assistant = AssistantAgent(
    name="Finance_Assistant",
    description=(
        "A Finance assistant that handles budget queries, "
        "expense reports, and financial summaries."
    ),
    system_message=f"""
You are Finance_Assistant in a sequential leave-processing workflow.

Calculate:
- Requested paid-leave days
- Daily employee cost
- Total estimated cost
- Budget remaining after the cost

Manager approval is required when:
- Estimated cost is at least ${APPROVAL_COST_THRESHOLD:,.2f}; or
- The request is a special leave or policy exception.

If approval is required, begin exactly with:

[From: Finance][Task: Calculate Leave Impact][Status: Approval Required][Next: Manager]

Include the calculation and this exact line:

Manager Approval Needed - Task Paused

If approval is not required, begin exactly with:

[From: Finance][Task: Calculate Leave Impact][Status: Complete][Next: End]

Do not approve or reject requests requiring human approval.
Do not place any greeting or Markdown before the structured header.
""",
    llm_config=llm_config,
)

employee = UserProxyAgent(
    name="Employee",
    description=(
        "Submits the initial leave request."
    ),
    human_input_mode="NEVER",
    code_execution_config=False,
    llm_config=False,
)

def select_next_speaker(last_speaker, groupchat):
    content = str(groupchat.messages[-1].get("content", "")).strip()

    try:
        labels = parse_labels(content)
    except ValueError as error:
        print(
            f"\nWORKFLOW STOPPED: {last_speaker.name} returned "
            f"an incorrectly formatted message.\n{error}"
        )
        return None

    if last_speaker is employee:
        if labels["next_agent"] != "HR":
            print("WORKFLOW STOPPED: Employee must assign the task to HR.")
            return None

        return hr_assistant

    if last_speaker is hr_assistant:
        valid_handoff = (
            labels["sender"] == "HR"
            and labels["status"] == "Complete"
            and labels["next_agent"] == "Finance"
            and "HR complete, please continue." in content
        )

        if not valid_handoff:
            print("WORKFLOW STOPPED: Invalid HR-to-Finance handoff.")
            return None

        return finance_assistant

    if last_speaker is finance_assistant:
        if approval_state["required"]:
            expected_handoff = (
                labels["sender"] == "Finance"
                and labels["status"] == "Approval Required"
                and labels["next_agent"] == "Manager"
            )

            if not expected_handoff:
                print(
                    "PROTOCOL WARNING: Finance supplied an incorrect routing "
                    "label. The deterministic approval rule overrode it."
                )

            return manager_approver

        valid_completion = (
            labels["sender"] == "Finance"
            and labels["status"] == "Complete"
            and labels["next_agent"] == "End"
        )

        if not valid_completion:
            print("WORKFLOW STOPPED: Invalid Finance completion message.")
            return None

    if last_speaker is manager_approver:
        valid_decision = (
            labels["sender"] == "Manager"
            and labels["status"] in {"Approved", "Rejected"}
            and labels["next_agent"] == "End"
        )

        if not valid_decision:
            print("WORKFLOW STOPPED: Invalid manager decision.")
            return None

        print(f"\nFinal manager decision: {labels['status']}")
        return None

    return None

workflow_chat = GroupChat(
    agents=[
        employee,
        hr_assistant,
        finance_assistant,
        manager_approver,
    ],
    messages=[],
    max_round=5,
    speaker_selection_method=select_next_speaker,
    allow_repeat_speaker=False,
    send_introductions=True,
)

workflow_manager = GroupChatManager(
    name="Workflow_Manager",
    groupchat=workflow_chat,
    llm_config=llm_config,
)

WORKFLOW_HEADER_PATTERN = re.compile(
    r"\[From:[^\]]+\]"
    r"\[Task:[^\]]+\]"
    r"\[Status:[^\]]+\]"
    r"\[Next:[^\]]+\]\s*",
    re.IGNORECASE,
)

def enforce_finance_protocol(sender, message, recipient, silent):
    """
    Deterministically apply Finance's workflow labels before its
    response reaches Workflow_Manager.
    """

    if isinstance(message, dict):
        original_content = str(message.get("content", ""))
    else:
        original_content = str(message)

    body = WORKFLOW_HEADER_PATTERN.sub(
        "",
        original_content,
        count=1,
    ).strip()

    pause_message = "Manager Approval Needed - Task Paused"

    if approval_state["required"]:
        header = (
            "[From: Finance]"
            "[Task: Calculate Leave Impact]"
            "[Status: Approval Required]"
            "[Next: Manager]"
        )

        if pause_message not in body:
            body = f"{body}\n\n{pause_message}"

    else:
        header = (
            "[From: Finance]"
            "[Task: Calculate Leave Impact]"
            "[Status: Complete]"
            "[Next: End]"
        )

        body = body.replace(pause_message, "").strip()

    corrected_content = f"{header}\n{body}"

    if isinstance(message, dict):
        corrected_message = dict(message)
        corrected_message["content"] = corrected_content
        return corrected_message

    return corrected_content

finance_assistant.register_hook(
    "process_message_before_send",
    enforce_finance_protocol,
)

employee.initiate_chat(
    recipient=workflow_manager,
    message="""
[From: Employee][Task: Submit Leave][Status: Ready][Next: HR]
I am requesting 40 hours of paid leave from September 14 through
September 18, 2026.

Available PTO balance: 80 hours
Daily employee cost: $320
Department leave-impact budget remaining: $10,000

Applicable policy: Requests of 12 days or fewer may be approved when
the employee has enough available PTO.
""".strip(),
)