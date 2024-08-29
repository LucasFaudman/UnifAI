from langchain.chains import NatBotChain
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, load_tools, AgentType

# Create a NatBotChain instance
llm = OpenAI(temperature=0.7)
# memory = ConversationBufferMemory()
# natbot = NatBotChain(
#     llm=llm, 
#     memory=memory,
#     objective="Find the email address of real estate agent David Faudman in CA."
#     )

# response = natbot.invoke("Whats the email of real estate agent David Faudman in CA?")

tools = load_tools(["google-search"], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

r = agent.run("What is the email of real estate agent David Faudman in CA?")
print(r)