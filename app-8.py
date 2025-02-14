import streamlit as st
import re
import numexpr as ne  # Fast numerical evaluation
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# Streamlit setup
st.set_page_config(page_title="Math Solver & Data Search")
st.title("Math Solver & Data Search")

# Sidebar API Key Input
groq_api_key = st.sidebar.text_input("Groq API KEY", value="", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key.")
    st.stop()

# Initialize LLM
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

# Wikipedia Tool
wiki_wrapper = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia",
    func=wiki_wrapper.run,
    description="Search Wikipedia for various information."
)

# Math Chain (LLM-based, for complex problems)
math_chain = LLMMathChain.from_llm(llm=llm)
math_tool = Tool(
    name="Math Solver",
    func=math_chain.run,
    description="Solves complex math problems. For simple math, use local calculation."
)

# Reasoning-based math prompt
math_prompt = """
You are a math expert. Solve the problem step-by-step and provide the final answer in this format:
`ANSWER: {final_numeric_answer}`.

Question: {question}
Solution:
"""

math_prompt_template = PromptTemplate(input_variables=["question"], template=math_prompt)

# LLM Chain for reasoning-based math
reasoning_chain = LLMChain(llm=llm, prompt=math_prompt_template)
reasoning_tool = Tool(
    name="Reasoning Tool",
    func=reasoning_chain.run,
    description="Solves text-based math and logic problems step-by-step."
)

# Initialize Agent
assistant_agent = initialize_agent(
    tools=[wiki_tool, math_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm your math chatbot! Ask me any question."}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Function to detect and solve simple math locally
def solve_math_locally(expression):
    try:
        return str(ne.evaluate(expression))
    except:
        return None

# Function to detect if input is a simple math expression
def is_math_expression(text):
    return bool(re.fullmatch(r"[\d\s\+\-\*/\(\)\.]+", text))

# User Input
question = st.chat_input("Ask your question...")

if question:
    with st.spinner("Thinking..."):
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        # Detect if it's a simple math expression
        if is_math_expression(question):
            response = solve_math_locally(question)
            if response:
                response = f"ANSWER: {response}"
            else:
                response = assistant_agent.run(question)
        else:
            # If it's a word-based problem, use LLM
            response = assistant_agent.run(question)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
