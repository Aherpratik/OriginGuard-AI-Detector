# OriginGuard-AI-Detector
🧠 AI Content &amp; Image Detector A lightweight tool to detect whether text or images are AI-generated or human-made. Built with Python and Streamlit, it uses advanced models like RoBERTa for text detection and deep learning for image analysis. 💡 Use Cases Academic and content integrity  Fake image detection  AI content moderation

Prerequisites

Python 3.10+

Suggested packages: streamlit, transformers, torch, scikit-learn, pillow / opencv-python.


Installation

git clone https://github.com/Aherpratik/OriginGuard-AI-Detector
cd OriginGuard-AI-Detector
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

For Running
streamlit run app.py

File Structure

originguard/
├─ app.py                      
├─ detectors/
│  ├─ text_detector.py         
│  ├─ paraphrase_guard.py      
│  └─ image_detector.py        
├─ configs/
│  ├─ defaults.yaml           
│  └─ logging.yaml
├─ tools/
│  ├─ batch_verify.py          
│  └─ eval.py                  
├─ data/                      
├─ docs/                       
└─ README.md

Evaluation

Text metrics: AUROC, F1 at selected thresholds, calibration (ECE).

Image metrics: accuracy, precision/recall, ROC curves.

Robustness checks: paraphrased text, OCR’d text, recompressed images, resized crops.

Reproducibility: fix seeds, pin package versions, log model commits/checkpoints.

Limitations & Ethics

False positives/negatives: No detector is perfect; always pair with human judgment.
Domain shift: Performance varies across languages, genres, and image domains.
Adversarial inputs: Heavy paraphrasing or image post-processing can degrade accuracy.
Privacy: Do not store user content unless explicitly opted-in.
Responsible use: Use results for guidance, not punitive decisions without review.

Citations & Acknowledgments

Text model: Hello-SimpleAI/ChatGPT Detector RoBERTa (credit to the original authors).
Thanks to open-source contributors in the Python/Streamlit ecosystem.
If you use OriginGuard in academic work, please cite this repo and the upstream model authors.

Below are some images from my project and in the end  there is video for complete demonstration of project

<img width="2544" height="1265" alt="image" src="https://github.com/user-attachments/assets/556c6630-ff05-429f-8930-d3959d0357bc" />
<img width="2544" height="1265" alt="image" src="https://github.com/user-attachments/assets/88624507-ffeb-4b73-a376-c3291d3b8be3" />
<img width="2544" height="1265" alt="image" src="https://github.com/user-attachments/assets/27bd9683-649a-461b-b73b-ebd967ecaf13" />
<img width="2544" height="1265" alt="image" src="https://github.com/user-attachments/assets/c96a1d0f-f2b4-4e4d-9afc-e14144288373" />


