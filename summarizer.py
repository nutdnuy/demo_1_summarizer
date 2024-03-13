from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate


########################## Prompt ##########################

map_prompt = """
Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:
"""

combine_paragraph_prompt = """
Your Role: Financial Analyst
Short basic instruction: Summarize economic outlook and investment documents from text enclosed within triple backquotes.
What you should do: Follow a structured approach to distill complex economic and investment documents into concise summaries. This includes reading thoroughly to understand the main arguments, identifying key points and arguments, and using your own words to outline a structured summary that captures the essence of the document.
Your Goal: Provide investors with a clear, concise summary that highlights the key points, arguments, and evidence from detailed economic outlook and investment documents. The summary should enable investors to quickly grasp the essential information and insights without having to navigate through the entire document.
Result: The summary should be organized with an introduction that includes the title of the original text, the author's name, and a brief overview of the main theme or argument. Body paragraphs should focus on specific main points or arguments, presented logically. The conclusion should reiterate the main arguments or points summarized. Use a formal and neutral tone throughout.
Constraint: Maintain objectivity and neutrality, avoiding personal opinions or biases. The summary should be significantly shorter than the original text, focusing only on essential arguments and points. Direct quotes should be used sparingly, with proper attribution.
Context: Given the complex nature of economic outlook and investment documents, which often contain detailed analyses, forecasts, and recommendations, your summary should help investors understand the key takeaways without oversimplifying or distorting the original document's content and intent.
```{text}```
PARAGRAPH SUMMARY:
"""

combine_bullet_prompt = """
Your Role: Financial Analyst
Short basic instruction: Summarize economic outlook and investment documents from text enclosed within triple backquotes.
What you should do: Follow a structured approach to read and understand the text, identify and organize key points, and summarize them effectively.
Your Goal: Provide concise, clear, and relevant summaries that aid investors in decision-making.
Result: Your summary should consist of bullet points, each representing a distinct idea or piece of information related to economics and investments. Use sub-bullets for details when necessary, ensuring each bullet is concise and uses keywords or phrases from the text.
Constraint: Limit your summary to the most critical information, avoiding overload. Keep each bullet point to one or two sentences and use parallel structure for readability. Focus on themes crucial for investment decisions, such as market trends, financial forecasts, risks, and opportunities.
Context: The document to be summarized is an economics outlook or investment document intended for investors. The summary should highlight essential points that affect investment decisions, maintaining accuracy and coherence with the original content.
```{text}```
BULLET POINT SUMMARY:
"""

map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
combine_paragraph_prompt_template = PromptTemplate(template=combine_paragraph_prompt, input_variables=["text"])
combine_bullet_prompt_template = PromptTemplate(template=combine_bullet_prompt, input_variables=["text"])

########################## Map-reduce ##########################

def map_reduce_paragraph(docs, model):
    summary_chain = load_summarize_chain(llm=model, chain_type='map_reduce',
                                         map_prompt=map_prompt_template,combine_prompt=combine_paragraph_prompt_template,
                                         # verbose=True
                                         )
    output = summary_chain.run(docs)
    return output

def map_reduce_bullet(docs, model):
    summary_chain = load_summarize_chain(llm=model, chain_type='map_reduce',
                                         map_prompt=map_prompt_template,combine_prompt=combine_bullet_prompt_template,
                                         # verbose=True
                                         )
    output = summary_chain.run(docs)
    return output

########################## Refine ##########################

def refine_paragraph(docs, model):
    chain = load_summarize_chain(llm=model, chain_type="refine",
                                 question_prompt=map_prompt_template,
                                 refine_prompt=combine_paragraph_prompt_template,
                                 return_intermediate_steps=False,
                                 input_key="input_documents", output_key="output_text",
                                 )
    output = chain({"input_documents": docs}, return_only_outputs=True)
    return output.get("output_text", "")

def refine_bullet(docs, model):
    chain = load_summarize_chain(llm=model, chain_type="refine",
                                 question_prompt=map_prompt_template,
                                 refine_prompt=combine_bullet_prompt_template,
                                 return_intermediate_steps=False,
                                 input_key="input_documents", output_key="output_text",
                                 )
    output = chain({"input_documents": docs}, return_only_outputs=True)
    return output.get("output_text", "")

########################## Translate ##########################

template_string = """
Your task is to translate the text found between the triple backticks below into Thai language.
Ensure the translation maintains a natural and fluent tone, and exclude the backticks from your response. Below is the text requiring translation:
```{text}```
Please adhere to the following guidelines in your translation:
- The translation should be natural and fluent, accurately reflecting the essence of the original text.
- Focus on translating only the section enclosed within the triple backticks; all other instructions should remain in English.
- Exclude the backticks in your translation, presenting a clear and focused response.
- Keep the format consistent: if the original text is in bullet points, maintain bullet points; if it's in paragraph form, keep it as a paragraph.
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

def translate_to_thai(docs, model):
    prompt = prompt_template.format_messages(text=docs)
    output = model(prompt)
    return output.content