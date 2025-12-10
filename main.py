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
    await minio.ensure_bucket()

    # --- Step 1: Initial Upload ---
    res = await cl.AskActionMessage(
        content="آیا می‌خواهید رزومه‌ها را از یک پوشه آپلود کنید؟",
        actions=[
            cl.Action(name="yes", payload={"value":"yes"}, label="بله, آپلود میکنم"),
            cl.Action(name="no", payload={"value":"no"}, label="نه, فعلا"),
        ]
    ).send()

    uploaded_keys = []
    
    if res and res.get("payload",{}).get("value") == "yes":
        files = await cl.AskFileMessage(
            content="Please drop your PDF resumes here.",
            accept=["application/pdf"],
            max_size_mb=20,
            max_files=10
        ).send()
        
        if files:
            msg = cl.Message(content=f"Uploading {len(files)} files...")
            await msg.send()
            
            tasks = [minio.upload_file(f.path, f.name) for f in files]
            await asyncio.gather(*tasks)
            uploaded_keys = [f.name for f in files]
            
            msg.content = f"✅ Done. Uploaded {len(files)} files."
            await msg.update()

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
        "all_files": uploaded_keys,
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
                
                # --- Handle Outputs ---
                if updates:
                    if "final_jd" in updates:
                        await cl.Message(content=f"**Generated Job Description:**\n\n{updates['final_jd']}").send()
                    
                    if "qa_answer" in updates:
                        await cl.Message(content=updates["qa_answer"]).send()

                # Handle Streaming Chat Messages
                for message_key in ["start_message", "jd_messages", "hiring_messages"]:
                    if updates:
                        if message_key in updates:
                            new_msgs = updates[message_key]
                            if isinstance(new_msgs, list) and new_msgs:
                                last_msg = new_msgs[-1]
                                if last_msg.type == "ai" and last_msg.content:
                                    await cl.Message(content=parser.invoke(last_msg)).send()

        # --- CYCLE FINISHED: Check state for Interrupts ---
        snapshot = await graph.aget_state(config)
        
        if snapshot.next and snapshot.tasks[0].interrupts:
            interrupt_val = snapshot.tasks[0].interrupts[0].value
            logger.debug(f"⏸️ DEBUG: Graph Interrupted. Value: {interrupt_val}") # DEBUG LOG
            
            # --- SCENARIO A: Graph wants a FILE UPLOAD ---
            if isinstance(interrupt_val, dict) and interrupt_val.get("type") == "compare_upload":
                
                bot_request = interrupt_val.get('msg', 'Please upload a file.')
                await cl.Message(content=f"🤖 {bot_request}").send()
                
                files = await cl.AskFileMessage(
                    content="Upload the PDF file(s) for comparison.",
                    accept=["application/pdf"],
                    max_size_mb=20,
                    max_files=5,
                    timeout=600 
                ).send()
                
                if files:
                    await cl.Message(content="📂 Uploading to MinIO...").send()
                    
                    tasks = [minio.upload_file(f.path, f.name) for f in files]
                    await asyncio.gather(*tasks)
                    
                    uploaded_keys = [f.name for f in files]
                    
                    # CRITICAL FIX: Ensure we have keys before resuming
                    if uploaded_keys:
                        logger.debug(f"▶️ DEBUG: Resuming with files: {uploaded_keys}")
                        # Resume the graph immediately
                        await run_graph_cycle(Command(resume=uploaded_keys))
                    else:
                        await cl.Message(content="❌ No file keys generated. Stopping.").send()
                else:
                    await cl.Message(content="❌ Upload cancelled or timed out.").send()
        
            elif interrupt_val in ["hiring_input", "router_node", "jd_node", "qa_input"]:
                return

    except Exception as e:
        logger.error(f"❌ ERROR in run_graph_cycle: {e}")
        await cl.Message(content=f"An error occurred: {str(e)}").send()