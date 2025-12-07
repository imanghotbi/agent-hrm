import asyncio
import os
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.mongodb import MongoDBSaver

from app.config.logger import logger
from app.services.minio_service import MinioHandler
from app.workflow.builder import build_graph
from app.config.config import config


async def main():
    minio = MinioHandler()
    await minio.ensure_bucket()

    logger.info("--- 🚀 Iran HR Agent System ---")
    
    # Input/Upload logic (Same as before, simplified for brevity here)
    choice = input("Do you want to upload resumes from a local folder? (y/n): ").strip().lower()
    if choice == 'y':
        folder_path = input("Enter path: ").strip()
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            tasks = [minio.upload_file(os.path.join(folder_path, f), f) for f in files]
            if tasks: 
                await asyncio.gather(*tasks)
                logger.info(f"Done. Uploaded {len(files)} files.")
            else:
                logger.info("No PDF files found in folder.")
        else:
            logger.info("❌ Folder not found. Skipping upload.")
    
    # Start Graph

    app = build_graph()
    with MongoDBSaver.from_conn_string(config.mongo_uri,config.mongo_db_name) as saver:
        graph = app.compile(checkpointer=saver)
        thread_config = {"configurable": {"thread_id": "unified_session_1"}}
        parser = StrOutputParser()

        inputs = {
            "all_files": [], 
            "start_message": [HumanMessage('Introduce yourself.')]
        }

        async for event in app.astream(inputs, config=thread_config):
            pass

        # Loop for Interrupts
        while True:
            snapshot = await app.aget_state(thread_config)
            if not snapshot.next: break
            
            if snapshot.tasks and snapshot.tasks[0].interrupts:
                interrupt_val = snapshot.tasks[0].interrupts[0].value
                
                # Handling input prompts based on interrupt value
                if isinstance(interrupt_val, dict) and interrupt_val.get("type") == "compare_upload":
                    print(f"\n🤖 {interrupt_val['msg']}")
                    val = input("👤 Path: ").strip()
                    uploaded_keys = []
                    if val.lower() not in ["exit", "quit"]:
                        # Simple Logic: Check if it's a folder or list of files
                        if os.path.isdir(val):
                            files = [os.path.join(val, f) for f in os.listdir(val) if f.endswith('.pdf')]
                        else:
                            files = [f.strip() for f in val.split(',')]
                        
                        # Upload them immediately
                        if files:
                            logger.info(f"Uploading {len(files)} files for comparison...")
                            tasks = [minio.upload_file(f, os.path.basename(f)) for f in files]
                            await asyncio.gather(*tasks)
                            # We return the MinIO keys (filenames) to the graph
                            uploaded_keys = [os.path.basename(f) for f in files]

                            async for event in app.astream(Command(resume=uploaded_keys), config=thread_config):
                                # Repeat printing logic
                                pass


                elif "qa_input" in str(interrupt_val):
                    val = input("\n❓ Question: ").strip()
                else:
                    val = input("\n👤 Response: ").strip()

                async for event in app.astream(Command(resume=val), config=thread_config):
                    pass
if __name__ == "__main__":
    asyncio.run(main())