import streamlit as st
import pandas as pd
import sys
import importlib.util  
from image_detector.image_detection import detect_ai_image
import tempfile, plotly.express as px, os



st.set_page_config(page_title="Multimodel Project by Pratik Asmi Eeshan", layout="wide")

# =============================================================================
# Dynamically importing the detector module because normally failed manier time
# =============================================================================

detector_path= os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'text_detection', 'detector.py'))

spec =importlib.util.spec_from_file_location("detector", detector_path)

detector= importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(detector)

    analyze_pdf_advanced = detector.analyze_pdf_advanced

except Exception as e:

    st.error(f" Failure to load the detector module: {e}")
    st.stop()

# ================================
# Adding  rewriting checker import
# ================================

rewriting_path =os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'text_detection', 'rewriting_checker.py'))
spec_checker = importlib.util.spec_from_file_location("rewriting_checker", rewriting_path)

rewriting_checker= importlib.util.module_from_spec(spec_checker)


spec_checker.loader.exec_module(rewriting_checker)




# Creating tabs in stremlit  for each features used
tab1, tab2, tab3 = st.tabs([
    "AI Text Detection",
    "Rewriting Checker",
    "AI Image Detector",
    
])


# =========================
# Tab 1: Detecting AI Text
# =========================
with tab1:
    st.title(" Detecting AI-Generated Text...!")

    st.markdown("Upload an PDF file to analyze its content for whether it is  AI-generated text or not, here we are using a RoBERTa model.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:


        with st.spinner("Analyzing the uploaded document.. this may take a few seconds "):
            try:
                df = analyze_pdf_advanced(uploaded_file, threshold=0.2)

                st.success("Analysis completed!")


                if "AI" in df["Label"].values:
                    st.warning("AI generated content detected in this document.")


                else:

                    st.success("No AI generated content detected. All chunks are  Human written.")
                st.dataframe(df.style.applymap(

                    lambda x: 'background-color: #ffcccc' if isinstance(x, float) and x > 0.2 else '',
                    subset=['AI_Probability']

                ))

                
                label_filter =st.selectbox("Filter by Label",  ["All", "Human", "AI"])
                if label_filter != "All":
                    df = df[df["Label"] == label_filter]

                st.markdown("###  Detailed Results")
                for i, row in df.iterrows():
                    with st.expander(f"Page {row['Page']} | Chunk {row['Chunk_Index']} | Label: {row['Label']} | AI Probability: {row['AI_Probability']}"):
                        st.write(row["Chunk_Text"])

                # here giving Summary count
                st.markdown("###  Summary of Detection")
                label_counts= df["Label"].value_counts()

                
                if "AI" in label_counts:
                    st.warning(f"AI generated content detected in {label_counts['AI']} this chunk(s).")
                else:
                    st.success("No AI generated content detected. All chunks are  Human written.")

                # Optional pie chart
                st.markdown("####  Visual Summary")

                st.plotly_chart(
                    {

                        "data": [

                            {
                                "values": label_counts.values,

                                "labels": label_counts.index,
                                "type": "pie",

                                "hole": 0.4,
                            }
                        ],
                        "layout": {"showlegend": True},
                    }
                )

                # Download of csv if user needed
                csv =df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Results for uploaded document as CSV", data=csv, file_name="ai_detection_results.csv", mime="text/csv")


                
                st.markdown("#### Confidence Weighted Visual Summary")

                # Calculating total number  of how much AI and Human detected
                total_chunks = len(df)
                ai_conf_sum = df[df["Label"] == "AI"]["AI_Probability"].sum()

                human_conf_sum = (1 - df[df["Label"] == "Human"]["AI_Probability"]).sum()

                # Normalizing it  to percentage
                total_conf= ai_conf_sum + human_conf_sum

                ai_percent =(ai_conf_sum / total_conf) * 100 if total_conf > 0 else 0

                human_percent = (human_conf_sum / total_conf) * 100 if total_conf > 0 else 0

               
                st.write(f"AI Weighted %: {ai_percent:.2f}, Human Weighted %: {human_percent:.2f}")

                # Plot the pie chart
                st.plotly_chart({
                    "data": [

                        {
                            "values": [human_percent, ai_percent],

                            "labels": ["Human (Confidence)", "AI (Confidence)"],
                            "type": "pie",

                            "hole": 0.4,
                        }
                    ],

                    "layout": {"showlegend": True}
                })

            except Exception as e:

                st.error(f"Something went wrong: {e}")
    else:
        st.info("Please upload a PDF file to get started....")


# ==================================
# Tab 2: Rewriting Checker feature 2
# ==================================
with tab2:
    st.title(" Rewriting AI Checker system")

     
    st.markdown("Paste your text below and we’ll detect whether it is generated by AI or not.")

    user_input = st.text_area(" Your Input Text")

    if st.button(" Check Rewriting Similarity"):

        result= rewriting_checker.check_rewriting(user_input)

        if result["status"] == "empty":

            st.warning("Please enter some text.")
        else:
            st.markdown("###  Similarity Score of text")


            st.write(f"*****{result['similarity']}***** (if 0 = No Similarity, 1 = Exact Match)")

            st.markdown(f"*****Interpretation:***** {result['label']}")

            
        if result["similarity"] < 0.3:


            st.success("Your text appears to be human written and not similar to known AI content generator.")


        elif result["similarity"]< 0.6:
            st.warning("""
            Your text shares some similarity with known AI generated phrases.

            This might mean it’s loosely inspired or rewritten from AI suggestions.

            If originality is critical, you might consider rephrasing some parts or you may suffer from AI voilations.
            """)


        else:
            st.error("""
            Your text is highly similar to known AI generated content.

            It may be directly copied or slightly modified.

            Strongly consider rewriting to ensure it's original and authentic you may suffer from AI voilations.
            """)




# ==========================================
# tab 3: Detecting whether it is an AI Image
# ===========================================

with tab3:
    st.title("AI Generated Image Detection")
    st.markdown("Upload an image to check whether it was AI generated or a real photo.")

    img_file =st.file_uploader("Choose an image", type=["jpg","jpeg","png"])

    if not img_file:

        st.info(" Please upload an image to get started....")
    else:
        
        suffix = os.path.splitext(img_file.name)[1]

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:


            tmp.write(img_file.read())

            tmp_path = tmp.name

        
        result= detect_ai_image(tmp_path)

        os.remove(tmp_path)

        ai_pct   = result["confidence"] * 100 if result["label"]=="AI Generated" else (100 - result["confidence"]*100)

        real_pct =100 - ai_pct

        
        if ai_pct > real_pct:
            st.warning(f" This image appears to be an ****AI-Generated**** ({ai_pct:.1f}% confidence).")

            st.write("It exhibits that synthetic artifacts overly smooth textures or unnatural details.")
        else:
            st.success(f" This appears to be a genuine photograph or naturally clicked ({real_pct:.1f}% confidence).")


            st.write("It shows very natural variation in color, texture, and lighting.")

        
        fig = px.pie(


            names=["Real", "AI-Generated"],

            values=[real_pct, ai_pct],
            hole=0.4,

            color_discrete_map={"Real":"#90ee90", "AI-Generated":"#ff6f61"}
        )


        st.plotly_chart(fig, use_container_width=True)



