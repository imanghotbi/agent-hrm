import asyncio
import io
import base64
from pdf2image import convert_from_bytes
from langchain_core.messages import HumanMessage
from app.config.config import config
from app.config.logger import logger
from app.services.llm_factory import LLMFactory
from app.services.minio_service import MinioHandler
from utils.prompt import OCR_PROMPT
from utils.extract_structure import save_token_cost

class OCRService:
    def __init__(self, node_name:str, session_id:str):
        self.semaphore = asyncio.Semaphore(config.ocr_workers)
        self.node_name = node_name
        self.session_id = session_id

    async def process_file(self, minio: MinioHandler, file_key: str) -> tuple[str, str | None]:
        async with self.semaphore:
            try:
                logger.info(f"🔹 [OCR START] {file_key}")
                pdf_bytes = await minio.download_file_bytes(file_key)
                
                # Run CPU-bound image conversion in a separate thread
                images = await asyncio.to_thread(convert_from_bytes, pdf_bytes, fmt='png', dpi=300)
                
                if not images:
                    logger.warning(f"⚠️ [OCR WARNING] No images converted for {file_key}")
                    return file_key, None

                text_result = ""
                llm = LLMFactory.get_model()

                for img in images:
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_b64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
                    
                    msg = HumanMessage(content=[
                        {"type": "text", "text": OCR_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ])
                    response = await llm.ainvoke([msg])
                    asyncio.create_task(save_token_cost(self.node_name, self.session_id, response))
                    text_result += response.content + "\n"

                return file_key, text_result

            except Exception as e:
                logger.error(f"❌ [OCR ERROR] {file_key}: {str(e)}", exc_info=True)
                return file_key, None