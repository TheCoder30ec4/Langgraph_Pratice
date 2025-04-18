from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

Reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ('system',
         """You are a professional email writer with expertise in crafting clear, simple, and effective emails.
         You specialize in breaking down complex ideas into easily understandable language, while also providing helpful explanations to ensure clarity and context. 
         Reflect on each email you write: evaluate its tone, structure, clarity, and whether it communicates the intended message effectively. 
         Suggest improvements where necessary to enhance simplicity, professionalism, and reader understanding."""),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

genration_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are a professional email writer with a talent for crafting clear, concise, and well-structured emails. 
         Your goal is to write emails in the simplest possible language while still sounding professional and informative. 
         You explain complex ideas in a way that is easy to understand, ensuring the message is accessible to any reader. 
         Generate emails that are direct, context-aware, and easy to follow, while maintaining a polite and appropriate tone for the given scenario."""),
        MessagesPlaceholder(variable_name="messages")
    ]
)


llm = ChatGroq(model="mistral-saba-24b")
reflection_chain = Reflection_prompt | llm
genration_chain = genration_prompt | llm 