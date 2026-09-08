"""
Personal Exam Evaluator
------------------------
A mobile-responsive Streamlit app that uses Google Gemini 2.5 Flash (Vision)
to compare a photographed official answer key against a photographed
handwritten student answer sheet, and produce a graded Markdown report.

Run locally:
    streamlit run app.py

Author: Senior Python & Cloud Developer (assisted)
"""

import io
import time

import streamlit as st
from PIL import Image, UnidentifiedImageError

# google-genai official SDK
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError


# --------------------------------------------------------------------------
# PAGE CONFIG & MOBILE-FIRST STYLING
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Personal Exam Evaluator",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Tighten top padding so content starts higher on small screens */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 760px;
    }

    /* Make all primary buttons big, bold, and tappable on mobile */
    div.stButton > button {
        height: 3.2em;
        font-size: 1.05em;
        font-weight: 600;
        border-radius: 12px;
    }

    /* Card-like containers for uploaders */
    .upload-card {
        background-color: rgba(120, 120, 120, 0.06);
        border: 1px solid rgba(120, 120, 120, 0.18);
        border-radius: 14px;
        padding: 0.9rem 1rem 0.4rem 1rem;
        margin-bottom: 1rem;
    }

    /* Header banner */
    .app-header {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
    }
    .app-header h1 {
        font-size: 1.7rem;
        margin-bottom: 0.1rem;
    }
    .app-header p {
        color: gray;
        font-size: 0.95rem;
        margin-top: 0;
    }

    /* Summary metrics tighten on small screens */
    @media (max-width: 480px) {
        div[data-testid="stMetric"] {
            padding: 0.3rem;
        }
    }

    hr {
        margin: 1.2rem 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>📝 Personal Exam Evaluator</h1>
        <p>Snap the answer key + your sheet, and let Gemini 2.5 Flash grade it for you.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# SIDEBAR — API KEY INPUT
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 Configuration")
    st.caption(
        "Your API key is used only for this session and is never stored or sent "
        "anywhere except directly to Google's Gemini API."
    )
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="Paste your API key here",
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )

    st.divider()
    st.markdown(
        "**Model used:** `gemini-2.5-flash`\n\n"
        "**How it works:**\n"
        "1. Upload the official answer key photo.\n"
        "2. Upload your handwritten answer sheet photo.\n"
        "3. Tap **Evaluate Answers**.\n"
        "4. Get an instant graded report."
    )
    st.divider()
    st.caption("Built with ❤️ using Streamlit + Google Gemini")


# --------------------------------------------------------------------------
# SYSTEM PROMPT — THE CORE GRADING INSTRUCTIONS FOR GEMINI
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are an extremely meticulous, fair, and experienced human exam evaluator
with decades of experience grading handwritten answer sheets against official
answer keys. You have excellent handwriting-recognition skills and can
confidently interpret messy, slanted, cursive, or inconsistent handwriting,
cross-outs, and re-written answers.

You will be given exactly two images, in this order:
- IMAGE 1: The OFFICIAL ANSWER KEY (the source of truth for correct answers).
- IMAGE 2: The STUDENT'S HANDWRITTEN ANSWER SHEET (the answers to be graded).

Your task, performed with extreme rigor and accuracy:

1. CAREFULLY read Image 1 and extract every question number along with its
   official correct answer (this may be a letter choice like A/B/C/D, a
   short word, a number, or a short phrase — extract exactly what is given).

2. CAREFULLY read Image 2 and extract every question number along with the
   student's handwritten answer for that same question number. Use your best
   judgment to decipher handwriting. If an answer is genuinely illegible or
   blank, mark the student's answer as "Illegible/Blank".

3. CROSS-EXAMINE each question number that appears in BOTH sheets:
   - Compare the official answer to the student's answer.
   - Be tolerant of minor variations that do not change meaning (case
     differences, extra whitespace, minor spelling variants, "A" vs "a)",
     synonyms for short-answer questions if the core concept is clearly the
     same). Use sound academic judgment — do not be overly pedantic, but do
     not be overly lenient either. When genuinely ambiguous, favor giving the
     student credit only if the core answer is unmistakably correct.
   - If a question number exists in the key but is missing, blank, or
     illegible in the student's sheet, mark it as Incorrect ("🔴") with the
     student's answer shown as "Not Attempted" or "Illegible".
   - If the student wrote an answer for a question number that does not
     exist in the official key, ignore that extra entry (do not include it
     in the final table).

4. Compute these summary statistics based ONLY on the questions found in the
   official answer key:
   - Total Questions (total count of questions in the official key)
   - Total Correct
   - Total Wrong
   - Accuracy Percentage (Total Correct / Total Questions * 100, rounded to
     one decimal place)

5. OUTPUT FORMAT — Follow this EXACT structure, in clean GitHub-flavored
   Markdown. Do not include any preamble, apology, disclaimer, or
   explanation of your process outside of this structure. Do not wrap the
   output in code fences.

---

## 📊 Evaluation Summary

| 📌 Metric | Value |
|---|---|
| 📋 Total Questions | **{total_questions}** |
| 🟢 Total Correct | **{total_correct}** |
| 🔴 Total Wrong | **{total_wrong}** |
| 🎯 Accuracy | **{accuracy}%** |

---

## 🧾 Detailed Question-by-Question Breakdown

| Question # | ✅ Correct Answer | ✍️ Your Answer | Status |
|---|---|---|---|
| 1 | <official answer> | <student answer> | 🟢 Correct |
| 2 | <official answer> | <student answer> | 🔴 Incorrect |
...(continue for every question found in the official key, in ascending
numerical order)...

---

### 💡 Quick Notes
- Add a short bullet list (2-4 bullets max) of brief, encouraging, and
  constructive observations — e.g. patterns of mistakes, topics to revisk,
  or praise for strong performance. Keep this concise and supportive.

---

IMPORTANT RULES:
- If Image 1 does not appear to be an answer key, or Image 2 does not appear
  to be a handwritten answer sheet, or either image is unreadable/blurry to
  the point where grading is not possible, DO NOT attempt to hallucinate an
  evaluation. Instead output exactly this single line and nothing else:
  "⚠️ UNABLE_TO_GRADE: One or both images could not be read clearly enough
  to perform an accurate evaluation. Please retake the photos with better
  lighting and focus, ensuring all question numbers and answers are visible."
- Never invent question numbers or answers that are not actually visible in
  the images.
- Always double-check your comparison logic before finalizing the table —
  accuracy is the single most important quality of your output.
- Respond only in the Markdown structure described above (or the single
  UNABLE_TO_GRADE line). No extra commentary before or after.
"""


# --------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------
def load_image_from_upload(uploaded_file):
    """Convert a Streamlit UploadedFile into a PIL Image (RGB)."""
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def call_gemini_evaluation(api_key: str, key_image: Image.Image, sheet_image: Image.Image) -> str:
    """
    Sends both images + the system prompt to Gemini 2.5 Flash and returns
    the raw Markdown text response. Raises exceptions on failure so the
    caller can handle them with user-friendly messages.
    """
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            "IMAGE 1 — OFFICIAL ANSWER KEY:",
            key_image,
            "IMAGE 2 — STUDENT'S HANDWRITTEN ANSWER SHEET:",
            sheet_image,
            "Now perform the evaluation exactly as instructed in the system prompt.",
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=4096,
        ),
    )

    # Defensive extraction of text
    text = getattr(response, "text", None)
    if not text:
        # Fallback: try to walk candidates/parts manually
        try:
            text = response.candidates[0].content.parts[0].text
        except Exception:
            text = None

    if not text or not text.strip():
        raise ValueError("Gemini returned an empty response. Please try again.")

    return text.strip()


# --------------------------------------------------------------------------
# MAIN UI — UPLOADERS
# --------------------------------------------------------------------------
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
st.subheader("1. Official Answer Key")
key_file = st.file_uploader(
    "Upload a clear photo of the official answer key",
    type=["png", "jpg", "jpeg"],
    key="key_uploader",
    label_visibility="collapsed",
)
if key_file is not None:
    st.image(key_file, caption="Official Answer Key Preview", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="upload-card">', unsafe_allow_html=True)
st.subheader("2. Your Answer Sheet")
sheet_file = st.file_uploader(
    "Upload a clear photo of your handwritten answer sheet",
    type=["png", "jpg", "jpeg"],
    key="sheet_uploader",
    label_visibility="collapsed",
)
if sheet_file is not None:
    st.image(sheet_file, caption="Your Answer Sheet Preview", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# EVALUATE BUTTON & LOGIC
# --------------------------------------------------------------------------
evaluate_clicked = st.button("🚀 Evaluate Answers", use_container_width=True, type="primary")

st.markdown("<hr>", unsafe_allow_html=True)
result_container = st.container()

if evaluate_clicked:
    # ---- Input validation ----
    if not api_key or not api_key.strip():
        st.warning("⚠️ Please enter your Google Gemini API Key in the sidebar before evaluating.")
    elif key_file is None or sheet_file is None:
        st.warning("⚠️ Please upload BOTH the official answer key and your answer sheet photos.")
    else:
        try:
            with st.spinner("🔎 Analyzing your answer sheets... please wait"):
                # Convert uploads to PIL images
                try:
                    key_image = load_image_from_upload(key_file)
                    sheet_image = load_image_from_upload(sheet_file)
                except UnidentifiedImageError:
                    st.error(
                        "❌ One of the uploaded files doesn't look like a valid image. "
                        "Please upload a clear PNG or JPG photo."
                    )
                    st.stop()

                # Small delay so the spinner feels intentional on very fast responses
                start_time = time.time()

                report_markdown = call_gemini_evaluation(api_key.strip(), key_image, sheet_image)

                elapsed = time.time() - start_time
                if elapsed < 0.5:
                    time.sleep(0.5 - elapsed)

            # ---- Handle the special "can't grade" signal from the model ----
            if report_markdown.startswith("⚠️ UNABLE_TO_GRADE"):
                with result_container:
                    st.warning(report_markdown)
                    st.info(
                        "💡 Tip: Make sure both photos are well-lit, in focus, and show "
                        "all question numbers and answers clearly without shadows or glare."
                    )
            else:
                with result_container:
                    st.success("✅ Evaluation complete!")
                    st.markdown(report_markdown)

                    st.download_button(
                        label="⬇️ Download Report as Markdown",
                        data=report_markdown,
                        file_name="exam_evaluation_report.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

        except ClientError as e:
            # Covers 4xx errors: bad API key, invalid request, quota issues, etc.
            status = getattr(e, "status_code", None)
            if status == 401 or status == 403:
                st.error(
                    "🔒 Authentication failed. Your Gemini API Key appears to be invalid "
                    "or unauthorized. Please double-check it in the sidebar."
                )
            elif status == 429:
                st.error(
                    "⏳ You've hit the free-tier rate limit for Gemini right now. "
                    "Please wait a minute and try again."
                )
            else:
                st.error(f"❌ The Gemini API rejected the request. Details: {str(e)}")

        except ServerError:
            st.error(
                "🌐 Google's Gemini servers seem to be having a temporary issue "
                "(server error / timeout). Please try again in a moment."
            )

        except TimeoutError:
            st.error(
                "⏱️ The request timed out. Please check your internet connection and try again."
            )

        except Exception as e:  # noqa: BLE001 - final safety net for a personal tool
            st.error(
                "❌ An unexpected error occurred while evaluating your answer sheets. "
                f"Details: {str(e)}"
            )

else:
    with result_container:
        st.info(
            "👆 Upload both images and tap **Evaluate Answers** to generate your report."
        )


# --------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.caption(
    "🔒 Privacy note: Images and your API key are processed only in-memory for this "
    "session and sent directly to Google's Gemini API. Nothing is stored on any server."
)
