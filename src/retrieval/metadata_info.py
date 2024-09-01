from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="paper_title",
        description="The title of the research paper",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="section_number",
        description="The section number to which the retrieved chunk of text belongs to",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="section_title",
        description="The section title to which the retrieved chunk of text belongs to",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="pages",
        description="The starting and the end page numbers of the chunk of text",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="para",
        description="The number of the paragraph in the selected pages",
        type="string or integer",
    ),
    AttributeInfo(
        name="file_path", 
        description="The file path of the research paper", 
        type="string"
    ),
]
document_content_description = "This is a research paper"

class MetadataInfo():
    def __init__(self, metadata_info=metadata_field_info, document_content_des=document_content_description) -> None:
        self.metadata_field_info = metadata_info
        self.document_content_description = document_content_des

    def addAttribute(self, name, desc, type):
        self.metadata_field_info.append(
            AttributeInfo(
                name=name, 
                description=desc, 
                type=type
            )
        )