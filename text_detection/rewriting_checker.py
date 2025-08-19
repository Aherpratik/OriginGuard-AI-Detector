# rewriting_checker.py

from sentence_transformers import SentenceTransformer, util

# Load the model only once
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Sample AI-generated base text (replace with your real reference text if needed)
AI_REFERENCE = """The integration of artificial intelligence into daily human activity has ushered in a new era of convenience, automation, and decision-making efficiency. This technological evolution is rapidly transforming industries from healthcare to finance."""

# Main function to compare user text to the reference
def check_rewriting(user_text, reference_text=AI_REFERENCE, threshold_high=0.8, threshold_low=0.5):
    if not user_text.strip():
        return {"status": "empty", "similarity": 0.0, "label": "Empty Input"}

    model = load_embedder()

    user_embedding = model.encode(user_text, convert_to_tensor=True)
    reference_embedding = model.encode(reference_text, convert_to_tensor=True)

    similarity = util.cos_sim(user_embedding, reference_embedding).item()

    # Interpret similarity score
    if similarity > threshold_high:
        label = "Highly Similar (Likely Rewritten)"
    elif similarity > threshold_low:
        label = "Moderately Similar (Possibly Rewritten)"
    else:
        label = "Low Similarity (Unlikely Rewritten)"

    return {
        "similarity": round(similarity, 3),
        "label": label,
        "status": "ok"
    }
