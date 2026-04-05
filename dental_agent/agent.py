from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse,AgentMiddleware


from dental_agent.config.settings import OPENAI_API_KEY, MODEL_NAME, TEMPERATURE, OPENAI_BASE_URL
from dental_agent.utils import sanitize_messages
from dental_agent.tools.csv_reader import (
    get_available_slots,
    get_patient_appointments,
    check_slot_availability,
    list_doctors_by_specialization,
)
from dental_agent.tools.csv_writer import (
    book_appointment,
    cancel_appointment,
    reschedule_appointment,
)

TOOLS = [
    get_available_slots,
    get_patient_appointments,
    check_slot_availability,
    list_doctors_by_specialization,
    book_appointment,
    cancel_appointment,
    reschedule_appointment,
]

SYSTEM_PROMPT = """You are a helpful dental appointment assistant. You help patients with:

1. Checking available appointment slots and doctor information
2. Booking new appointments
3. Cancelling existing appointments
4. Rescheduling appointments

## Available Specializations
general_dentist, oral_surgeon, orthodontist, cosmetic_dentist,
prosthodontist, pediatric_dentist, emergency_dentist

## Date Format
Always use M/D/YYYY H:MM format — e.g. 5/10/2026 9:00

## Booking Rules
- Always call check_slot_availability before booking to confirm the slot is free
- If a slot is taken, call get_available_slots to suggest alternatives
- Always confirm cancellations before executing them
- Ask for one missing detail at a time — don't overwhelm the user
"""





class SanitizeMessagesMiddleware(AgentMiddleware):
    def wrap_tool_call(self, tool_call, next_handler):
        return next_handler(tool_call)

    def before_llm_call(self, state: dict) -> dict:
        sanitized = sanitize_messages(state["messages"])
        return {**state, "messages": sanitized}

llm = ChatOpenAI(api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                model=MODEL_NAME,
                temperature=TEMPERATURE)

dental_graph = create_agent(model=llm,
                            tools=TOOLS,
                            system_prompt=SYSTEM_PROMPT,
                            middleware= [SanitizeMessagesMiddleware()],
                            )