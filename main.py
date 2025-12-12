import chainlit as cl
import asyncio
from pymongo import MongoClient

# Graph & Service Imports
from langgraph.types import Command
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

# Your existing project imports
from app.services.minio_service import MinioHandler
from utils.helper import upload_resume_to_minio
from app.workflow.builder import build_graph
from app.config.config import config
from app.config.logger import logger

# Initialize services
mongo_client = MongoClient(config.mongo_uri)
parser = StrOutputParser()
minio = MinioHandler()
checkpointer = MongoDBSaver(mongo_client)

@cl.on_chat_start
async def start():
    logger.info("🚀 Session Started")

    # --- Step 1: check bucket exist---
    await minio.ensure_bucket(config.minio_resume_bucket)
    await minio.ensure_bucket(config.minio_compare_bucket)

    # --- Step 2: Build & Compile Graph ---
    builder = build_graph()
    graph = builder.compile(checkpointer=checkpointer)
    
    # Generate a unique thread ID for this user session
    thread_id = f"session_{cl.user_session.get('id')}"
    thread_config = {"configurable": {"thread_id": thread_id}}
    
    cl.user_session.set("graph", graph)
    cl.user_session.set("config", thread_config)

    # --- Step 3: Start the Graph ---
    initial_inputs = {
        "session_id": thread_id,
        "all_files": [],
        "start_message": [HumanMessage(content='Introduce yourself.')]
    }
    
    await run_graph_cycle(initial_inputs)

@cl.on_message
async def on_message(message: cl.Message):
    # Resume the graph with the text provided by the user
    # Note: If the graph is waiting for a FILE interrupt, sending text might fail 
    # depending on your graph logic.
    await run_graph_cycle(Command(resume=message.content))

async def run_graph_cycle(input_data):
    """
    Main loop to run the graph until it stops or hits an interrupt.
    """
    graph = cl.user_session.get("graph")
    config = cl.user_session.get("config")
    
    logger.debug(f"🔄 DEBUG: Starting cycle with input: {input_data}") # DEBUG LOG

    try:
        # stream_mode="updates" allows us to react to node completion
        async for event in graph.astream(input_data, config=config, stream_mode="updates"):
            
            for node_name, updates in event.items():
                logger.debug(f"📍 DEBUG: Node '{node_name}' finished.") # DEBUG LOG
                msg = cl.Message(content=f"...رزومه‌ها در حال بررسی هستند")

                if "process_batch_subgraph" in node_name or "compare_process" in node_name:
                    msg.content = "رزومه در حال ocr هستند"
                    await msg.update()

                if "save_results" in node_name:
                    msg.content = "فرآیند ارزیایی و نمره دهی به اتمام رسید."
                    await msg.update() 
                
                if "load_and_shard" in node_name:
                        await msg.send()
                        
                # --- Handle Outputs ---
                if updates:
                    if "final_jd" in updates:
                        await cl.Message(content=f"**شرح شغلی ایجاد شده:**\n\n{updates['final_jd']}").send()
                    
                    if "qa_answer" in updates:
                        await cl.Message(content=updates["qa_answer"]).send()

                    if "top_candidate" in updates:
                        await cl.Message(content=updates["top_candidate"]).send()

                        
                # Handle Streaming Chat Messages
                for message_key in ["start_message", "jd_messages", "hiring_messages", "comparison_context","compare_qa_answer"]:
                    if updates:
                        if message_key in updates:
                            new_msgs = updates[message_key]
                            if isinstance(new_msgs, list) and new_msgs:
                                last_msg = new_msgs[-1]
                                if last_msg.type == "ai" and last_msg.content:
                                    await cl.Message(content=parser.invoke(last_msg)).send()
                            elif isinstance(new_msgs, str) and new_msgs:
                                await cl.Message(content=parser.invoke(new_msgs)).send()

        # --- CYCLE FINISHED: Check state for Interrupts ---
        snapshot = await graph.aget_state(config)
        
        if snapshot.next and snapshot.tasks[0].interrupts:
            interrupt_val = snapshot.tasks[0].interrupts[0].value
            logger.debug(f"⏸️ DEBUG: Graph Interrupted. Value: {interrupt_val}") # DEBUG LOG
            
            # --- SCENARIO A: Graph wants a FILE UPLOAD ---
            if isinstance(interrupt_val, dict) and (interrupt_val.get("type") == "compare_upload" or interrupt_val.get("type") == "upload_resume"):
                
                res = await cl.AskActionMessage(
                content="آیا می‌خواهید رزومه‌ها را از یک پوشه آپلود کنید؟",
                actions=[
                    cl.Action(name="yes", payload={"value":"yes"}, label="بله, آپلود میکنم"),
                    cl.Action(name="no", payload={"value":"no"}, label="نه, فعلا"),
                ]).send()

                if res and res.get("payload",{}).get("value") == "yes":
                    files = await cl.AskFileMessage(
                        content="رزومه‌های مورد نیاز را آپلود کنید.",
                        accept=["application/pdf"],
                        max_size_mb=20,
                        max_files=20,
                        timeout=600 
                    ).send()

                    if files:
                        msg = cl.Message(content=f"در حال آپلود {len(files)} فایل ...")
                        await msg.send()
                        
                        bucket_name = interrupt_val.get("bucket_name")
                        uploaded_keys = await upload_resume_to_minio(files , bucket_name)
                        if uploaded_keys:
                            msg.content = f"✅ آپلود {len(files)} فایل انجام شد."
                            await msg.update()
                            logger.debug(f"▶️ DEBUG: Resuming with files: {uploaded_keys}")
                            # Resume the graph immediately
                            await run_graph_cycle(Command(resume=uploaded_keys))
                        else:
                            await cl.Message(content="❌ خطایی رخ داده است").send()
                    else:
                        await cl.Message(content="❌ آپلود کنسل یا ارتباط قطع شد.").send()
                else:
                    msg = cl.Message(content=f"رزومه های ذخیره شده بررسی می شود.")
                    await msg.send()
                    await run_graph_cycle(Command(resume=[]))

            elif interrupt_val in ["hiring_input", "router_node", "jd_node", "qa_input"]:
                return

    except Exception as e:
        logger.error(f"❌ ERROR in run_graph_cycle: {e}")
        await cl.Message(content=f"خطایی رخ داده است: {str(e)}").send()