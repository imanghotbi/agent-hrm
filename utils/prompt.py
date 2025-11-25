# -- PROMPTS --

OCR_PROMPT = (
    "You are a high-accuracy OCR engine. "
    "Transcribe the text from these resume pages exactly as it appears. "
    "Preserve the structure using Markdown headers (#, ##) and bullet points. "
    "If the text is in Persian, transcribe it in Persian. "
    "DO NOT summarize. Output only the raw text!! "
    "Pay close attention to the numbers so that nothing is missed or extracted incorrectly."
)

STRUCTURE_PROMPT_TEMPLATE = (
    "You are an expert CV parser regarding Iranian culture and Resume Styles. "
    "Extract details from the following Resume text into the structured JSON format. "
    "Write in Persian as much as possible in the fields that can be written. "
    "For Military Service (Sarbazi), look for keywords like 'Payan Khedmat', 'Maafiat', 'Mashmool'. "
    "Strictly adhere to the schema.\n\n"
    "RESUME TEXT:\n{raw_text}"
)
