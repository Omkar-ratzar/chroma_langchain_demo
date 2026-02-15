from langchain_ollama.llms import OllamaLLM # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from vector import retriver
model = OllamaLLM(model="llama3.2")
template='''
You are an expert in answering questions about a restraunt

Here are some reviews {reviews}
Here's the question that you have to answer: {question}'''

prompt=ChatPromptTemplate.from_template(template)
chain = prompt | model

while(True):
    print("\n\n________________________________")
    question=input("Ask, press Q to exit: ")
    print()
    if(question=="Q"):
        break

    reviews=retriver.invoke(question)
    result = chain.invoke({"reviews":reviews,"question":question})
    print(result)
