from fastapi import FastAPI, HTTPException, Query, Response, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, validator
import pandas as pd
import re
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from ibm_watsonx_ai.foundation_models import Model

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())
    
# =======================================================================
# ===============================Model 1=================================
# =======================================================================

@app.get("/model1", response_class=HTMLResponse)
async def read_model1():
    with open("./pages/model1.html", "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())
    
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from ibm_watsonx_ai.foundation_models import Model


# Load AraBERT model and tokenizer
model_name = "aubmindlab/bert-base-arabertv2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate embedding for text
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# Load poem embeddings
npy_dir = './processed_files/embeddings'
all_poem_embeddings = []
for file in sorted(os.listdir(npy_dir)):
    if file.endswith('.npy'):
        embeddings = np.load(os.path.join(npy_dir, file))
        all_poem_embeddings.append(embeddings)
combined_embeddings = np.vstack(all_poem_embeddings)

# Load poem data
csv_path = './content/Arabic Poem Comprehensive Dataset (APCD).csv'
full_poem_df = pd.read_csv(csv_path)

# Define metadata columns
metadata_columns = ['العصر', 'الشاعر', 'البحر']

# Define meter rules
meter_rules = {
    'الطويل': {
        'description': 'البحر الطويل يتكوّن من تفعيلتين في كل شطر: فَعُولُن مَفَاعِيلُن فَعُولُن مَفَاعِيلُن',
        'pattern': 'فَعُولُن مَفَاعِيلُن فَعُولُن مَفَاعِيلُن',
        'prosody': '– ᴗ – – | – ᴗ ᴗ – – | – ᴗ – – | – ᴗ ᴗ – –',
        'example': 'لَعَمْرُكَ مَا الأَيَّامُ إِلَّا مَعَارَةٌ \n فَمَا اسْتَطَلَتْ أَيَّامَهَا رُبَّ طُولِ',
        'prosodic_example': 'لَ عَمْ رُ كَ | مَا الْ أَيَّا مُ إِلْ | لَا مَعَارَةٌ | فَمَا اسْطَ طَلَتْ'
    },
    'المنسرح': {
        'description': 'البحر المنسرح يتكوّن من تفعيلتين: مُسْتَفْعِلُن مَفْعُولاتُ مُسْتَفْعِلُن',
        'pattern': 'مُسْتَفْعِلُن مَفْعُولاتُ مُسْتَفْعِلُن',
        'prosody': '– ᴗ – ᴗ – | – ᴗ – – ᴗ – | – ᴗ – ᴗ –',
        'example': 'إنْ قُلْتُها فَاهْتَمِمْ لها إنَّها \n نَفْسُك وَالْعُمْرُ تَارِكُه لَا تُرِيدُ',
        'prosodic_example': 'إنْ قُلْ | تُها فَا | هْتَمِمْ لها | إنَّها نَفْ | سُك وَالْعُمْ | رُ تَارِ كُه'
    },
    'المتقارب': {
        'description': 'البحر المتقارب يتكوّن من تفعيلتين في كل شطر: فَعُولُن فَعُولُن فَعُولُن فَعُولُن',
        'pattern': 'فَعُولُن فَعُولُن فَعُولُن فَعُولُن',
        'prosody': '– ᴗ – – | – ᴗ – – | – ᴗ – – | – ᴗ – –',
        'example': 'قِفَا نَبْكِ مِن ذِكْرَى حَبِيبٍ ومَنزِلِ \n بِسِقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ',
        'prosodic_example': 'قِ فَا نَبْ كِ | مِنْ ذِكْ رَى حَ | بِي بِ ن وَمَن | زِلِ'
    },
    'الخفيف': {
        'description': 'البحر الخفيف يتكوّن من تفعيلتين في كل شطر: فَاعِلاتُن مُسْتَفْعِلُن فَاعِلاتُن',
        'pattern': 'فَاعِلاتُن مُسْتَفْعِلُن فَاعِلاتُن',
        'prosody': 'ᴗ – – ᴗ – | ᴗ – ᴗ – ᴗ – | ᴗ – – ᴗ –',
        'example': 'إِنَّ الدُّنْيا مَطْلَبُها كَبِيرٌ \n وَقَلِيلٌ مَنْ يُدْرِكُ الْمُرَادَ',
        'prosodic_example': 'إِنْ نَ دْ دُنْ | يَا مَ طْ لَ بُهَا | كَ بِي رٌ وَ قَ | لِي لٌ مَنْ'
    },
    'الكامل': {
        'description': 'البحر الكامل يتكوّن من ثلاث تفعيلات في كل شطر: مُتَفَاعِلُن مُتَفَاعِلُن مُتَفَاعِلُن',
        'pattern': 'مُتَفَاعِلُن مُتَفَاعِلُن مُتَفَاعِلُن',
        'prosody': 'ᴗ – ᴗ – ᴗ – | ᴗ – ᴗ – ᴗ – | ᴗ – ᴗ – ᴗ –',
        'example': 'سَعى زَيدٌ لِيَلقَى عَيشَةً أُخْرى \n وَعادَ كَمَا كانَ لا شيءَ يُبْقِيهِ',
        'prosodic_example': 'سَ عَى زَيْ دٌ | لِ يَ لْ قَى | عَيْ شَةً أُخْ | رَى وَ عَا دَ'
    },
    'السريع': {
        'description': 'البحر السريع يتكوّن من تفعيلتين في كل شطر: مُسْتَفْعِلُن مُسْتَفْعِلُن فاعِلُن',
        'pattern': 'مُسْتَفْعِلُن مُسْتَفْعِلُن فاعِلُن',
        'prosody': '– ᴗ – ᴗ – | – ᴗ – ᴗ – | – ᴗ –',
        'example': 'لَمْ يَبْقَ مِنْهُمْ أَحَدٌ تَقَدَّمَا \n وَلا ظَلَّتْ فِي يَوْمِهِمْ قِسْمَةٌ',
        'prosodic_example': 'لَمْ يَبْ | قَ مِنْ هُمْ | أَ حَدٌ تَ قَ | دَّمَا وَ لا'
    },
    'الوافر': {
        'description': 'البحر الوافر يتكوّن من تفعيلتين في كل شطر: مُفَاعَلَتُن مُفَاعَلَتُن مُفَاعَلَتُن',
        'pattern': 'مُفَاعَلَتُن مُفَاعَلَتُن مُفَاعَلَتُن',
        'prosody': '– ᴗ ᴗ – ᴗ ᴗ – ᴗ ᴗ –',
        'example': 'أُقِلِّي اللَّوْمَ عَاذِلَ وَالعِتَابَا \n وَقُولِي إِنْ أَصَبْتُ لَقَدْ أَصَابَا',
        'prosodic_example': 'أُقِلْ لِي | اللَّوْ مَ عَا | ذِلَ وَ الْ عِ | تَا بَا وَ قُو'
    },
    'الرجز': {
        'description': 'البحر الرجز يتكوّن من تفعيلة واحدة تُكرّر: مُسْتَفْعِلُن مُسْتَفْعِلُن مُسْتَفْعِلُن',
        'pattern': 'مُسْتَفْعِلُن مُسْتَفْعِلُن مُسْتَفْعِلُن',
        'prosody': '– ᴗ – ᴗ – | – ᴗ – ᴗ – | – ᴗ – ᴗ –',
        'example': 'أَسِيرُ مِن أَمَامِكَ طَرِيقًا \n وَأَخَطُو فِي قَدَرِي وَلَا أَرَى',
        'prosodic_example': 'أَسِيرُ | مِن أَمَا مِ كَ | طَرِي قًا'
    },
    'البسيط': {
        'description': 'البحر البسيط يتكوّن من تفعيلتين في كل شطر: مُسْتَفْعِلُن فَاعِلُن مُسْتَفْعِلُن فَاعِلُن',
        'pattern': 'مُسْتَفْعِلُن فَاعِلُن مُسْتَفْعِلُن فَاعِلُن',
        'prosody': '– ᴗ – ᴗ – | – ᴗ – – | – ᴗ – ᴗ – | – ᴗ – –',
        'example': 'إِذَا غَامَرْتَ فِي شَرَفٍ مَرُومِ \n فَلَا تَقْنَعْ بِمَا دُونَ النُّجُومِ',
        'prosodic_example': 'إِذَا غَ | مَرْتَ فِي | شَرَ فٍ مَ | رُومِ فَ لَا'
    },
    'الرمل': {
        'description': 'البحر الرمل يتكوّن من تفعيلتين في كل شطر: فَاعِلَاتُن فَاعِلَاتُن فَاعِلَاتُن',
        'pattern': 'فَاعِلَاتُن فَاعِلَاتُن فَاعِلَاتُن',
        'prosody': 'ᴗ – – ᴗ – | ᴗ – – ᴗ – | ᴗ – – ᴗ –',
        'example': 'يَا لَهَا مِنْ فِتْنَةٍ مَا أَشْبَهَتْ \n أَحْلَامَنَا فِي كُلِّ لَحْنٍ مُطْرِبِ',
        'prosodic_example': 'يَا لَ هَا | مِنْ فِتْ نَةٍ | مَا أَشْ بَهَتْ'
    },
    'المجتث': {
        'description': 'البحر المجتث يتكوّن من تفعيلتين: مُسْتَفْعِلُن فَاعِلَاتُن',
        'pattern': 'مُسْتَفْعِلُن فَاعِلَاتُن',
        'prosody': '– ᴗ – ᴗ – | ᴗ – – ᴗ –',
        'example': 'إِذَا تَرَى بِنَا حَالَةَ الصَّبْرِ \n نَبْكِي عَلَى مَا فَاتَ مِنْ أَمَلِ',
        'prosodic_example': 'إِذَا تَرَ | بِنَا حَا | لَةَ الصَّبْ رِ'
    },
    'المديد': {
        'description': 'البحر المديد يتكوّن من تفعيلتين: فَاعِلَاتُن فَاعِلُن فَاعِلَاتُن',
        'pattern': 'فَاعِلَاتُن فَاعِلُن فَاعِلَاتُن',
        'prosody': 'ᴗ – – ᴗ – | ᴗ – ᴗ – | ᴗ – – ᴗ –',
        'example': 'شَرِبْنَا عَلَى ذِكْرِ الْحَبِيبِ مِدَامَةً \n فَنَاجَيْتُ نَفْسِي وَهِيَ نَفْسُ الغَرَامِ',
        'prosodic_example': 'شَرِبْ نَا | عَ لَى ذِ كْ | رِ الْ حَ بي ب'
    },
    'الهزج': {
        'description': 'البحر الهزج يتكوّن من تفعيلة واحدة تُكرّر: مَفَاعِيلُن مَفَاعِيلُن',
        'pattern': 'مَفَاعِيلُن مَفَاعِيلُن',
        'prosody': '– ᴗ ᴗ – – | – ᴗ ᴗ – –',
        'example': 'يا دارُ ما فَعَلَتْ بِكِ الأَيَّامُ؟ \n وَأَينَ سَكَنُ الدِّيارِ وَالسُّرَّامُ؟',
        'prosodic_example': 'يا دَا رُ | ما فَعَ لَتْ'
    },
    'المتدارك': {
        'description': 'البحر المتدارك يتكوّن من تفعيلة واحدة تُكرّر: فَاعِلُن فَاعِلُن',
        'pattern': 'فَاعِلُن فَاعِلُن',
        'prosody': 'ᴗ – ᴗ – | ᴗ – ᴗ –',
        'example': 'جاءَتْ مَعَ الصُّبْحِ أَمْشَاطُهَا \n وَسِرْنَا بَيْنَ قُصُورٍ وَظِلالٍ',
        'prosodic_example': 'جاءَ تْ | مَعَ الصُّ بْ حِ'
    },
    'المقتضب': {
        'description': 'البحر المقتضب يتكوّن من تفعيلتين: مُفْعِلَتُن فَاعِلَاتُ',
        'pattern': 'مُفْعِلَتُن فَاعِلَاتُ',
        'prosody': '– ᴗ – ᴗ – | ᴗ – – ᴗ –',
        'example': 'حَدَثَنِي قَائِلٌ عَاقِلٌ \n إِذَا قُمْتَ لَا تَتَأَخَّرُ أَكْثَرَ',
        'prosodic_example': 'حَدَ ثَ نِي | قَا إِ لٌ عَا'
    },
    'المضارع': {
        'description': 'البحر المضارع يتكوّن من تفعيلتين: مَفَاعِلَتُن فَاعِلَاتُن',
        'pattern': 'مَفَاعِلَتُن فَاعِلَاتُن',
        'prosody': '– ᴗ ᴗ – ᴗ – | ᴗ – – ᴗ –',
        'example': 'أَيُّهَا السَّائِلُ عَنِّي \n إِنَّنِي فِي صَحْوِ رُوحِي',
        'prosodic_example': 'أَيُّ هَا | السَّائِ لُ عَنِّي'
    },
    'شعر التفعيلة': {
        'description': 'شعر التفعيلة هو شعر حديث يلتزم بتفعيلة واحدة ولكنه غير ملتزم بالأوزان التقليدية الكاملة.',
        'pattern': 'يعتمد على تكرار التفعيلة بشكل حر',
        'prosody': 'حر وغير منتظم',
        'example': 'أَنتِ وَحْدَكِ الوَطَنُ \n وَأَنتِ الهَوَى يَغْمُرُ القَلْبَ صَبَا',
        'prosodic_example': '—'
    },
    'الدوبيت': {
        'description': 'الدوبيت هو نوع من الشعر الفارسي الذي انتقل إلى الأدب العربي، ويعتمد على الرباعية.',
        'pattern': 'نمط رباعي',
        'prosody': 'غير ثابت',
        'example': 'شَاعَتِ الأَنْوَارُ فِي كُلِّ مَكَانٍ \n وَأَرَاكَ فِي الأَكْوَانِ أَرْقَى مِن مَلَكِ',
        'prosodic_example': '—'
    },
    'موشح': {
        'description': 'الموشح هو نوع من الشعر الأندلسي الذي يتميز بتقسيم الأبيات إلى أقسام متساوية ولحن غنائي.',
        'pattern': 'نمط غنائي متعدد القوافي',
        'prosody': 'غير ثابت',
        'example': 'يا لَيْلَ الصَّبِّ مَتَى غَدَاتُهُ؟ \n أَقِيَامُ السَّاعَةِ مَوْعِدُهُ؟',
        'prosodic_example': '—'
    },
    'السلسلة': {
        'description': 'السلسلة هو نوع من الشعر يتميز بالتكرار والتدفق المستمر للأبيات.',
        'pattern': 'لا يوجد نمط معين',
        'prosody': 'غير ثابت',
        'example': 'لا توجد أمثلة معروفة.',
        'prosodic_example': '—'
    },
    'المواليا': {
        'description': 'المواليا هو نوع من الشعر الشعبي يتكوّن من أربعة أشطر وقافية موحدة.',
        'pattern': 'قافية موحدة بين الأشطر',
        'prosody': 'غير ثابت',
        'example': 'قُلْتُ لِلشَّمْسِ أَخْبِرِينِي عَنْ هَوَاهَا؟ \n فَأَجَابَتْ: لا يَزَالُ فِي النُّورِ نَسِيمُهَا',
        'prosodic_example': '—'
    },
    'شعر حر': {
        'description': 'الشعر الحر هو نوع من الشعر الذي لا يلتزم بالأوزان والقوافي التقليدية.',
        'pattern': 'حر بلا قافية أو وزن',
        'prosody': 'غير منتظم',
        'example': 'الشمسُ تشرقُ ثم تغيبُ \n كَأنها تشكو صمتَ النهارِ.',
        'prosodic_example': '—'
    },
    'عامي': {
        'description': 'الشعر العامي هو الشعر الذي يُكتب باللهجة العامية غير الفصحى.',
        'pattern': 'حسب اللهجة المحلية',
        'prosody': 'غير ثابت',
        'example': 'يا زِين مَكَانِك حَولي \n تْطُول اللْيَالِي بْدُونِك',
        'prosodic_example': '—'
    }
}


# Function to get meter rules based on meter name
def get_meter_rules(meter_name):
    return meter_rules.get(meter_name, None)

# Function to provide Arabic prosody rules
def provide_arudi_rules():
    prosody_rules = """
    ### القواعد الأساسية للتقطيع العروضي:
    - **الحركة**: أي صوت قصير (مثل "بَ") يُعبّر عنه بالرمز `/`.
    - **السكون**: الحرف الساكن (مثل "دْ") يُعبّر عنه بالرمز `–`.
    - **الحروف الطويلة (الألف والواو والياء الطويلة)**: الحروف الطويلة تُعبّر عنها بالرمز `–` (مثل "فَا").
    ### الوحدات العروضية:
    - **السبب الخفيف**: يتكون من حركة متبوعة بسكون، مثل "لَمْ" = `/ –`.
    - **السبب الثقيل**: يتكون من حركتين، مثل "عَلَى" = `/ /`.
    - **الوتد المجموع**: يتكون من حركتين متبوعتين بسكون، مثل "وَاحِدْ" = `/ – /`.
    - **الوتد المفروق**: يتكون من حركة، سكون، ثم حركة، مثل "مَوْلًى" = `/ – /`.
    ### أمثلة على التفعيلات:
    - **مُسْتَفْعِلُن**: `/ – / – /`.
    - **فَاعِلُن**: `/ – / –`.
    - **مَفَاعِيلُن**: `/ – – / –`.
    ### أمثلة على بحور الشعر:
    - **البحر الطويل**: فَعُولُن مَفَاعِيلُن فَعُولُن مَفَاعِيلُن.
    - **البسيط**: مُسْتَفْعِلُن فَاعِلُن مُسْتَفْعِلُن فَاعِلُن.
    """
    return prosody_rules

# Function to create a cleaned-up validation prompt and include Arabic prosody rules
def create_clean_validation_prompt(generated_poem, meter_name):
    meter_rules = get_meter_rules(meter_name)
    if meter_rules:
        arudi_rules = provide_arudi_rules()
        validation_prompt = f"""
{arudi_rules}

### مهمتك هي التأكد من أن الأبيات التالية تتبع القواعد الشعرية الخاصة بالبحر "{meter_rules['description']}" ({meter_name}).
إذا وجدت أي خلل في الوزن، يرجى اقتراح تعديلات لتحسين الأبيات.

### الأبيات:
{generated_poem}

### قواعد البحر الشعري ({meter_name}):

- التفعيلات: {meter_rules['pattern']}
- التقطيع العروضي: {meter_rules['prosody']}
- مثال: "{meter_rules['example']}"

يرجى مراجعة الأبيات والتأكد من أنها تتبع التفعيلات الخاصة بـ "{meter_rules['description']}". إذا كانت الأبيات صحيحة، يرجى تأكيد ذلك. إذا لم تكن كذلك، يرجى اقتراح التعديلات.
"""
        return validation_prompt
    else:
        return f"Meter '{meter_name}' not found."

# Fetch credentials
def get_credentials():
    return {
        "url": "https://eu-de.ml.cloud.ibm.com",
        "apikey": "bkIJnxA7mDVQPkbqGnjbeWQZnNjqXIoM9nmP5f7tQ_jU"
    }

# Initialize the model using the IBM Watsonx AI SDK
credentials = get_credentials()
model_id = "sdaia/allam-1-13b-instruct"
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 900,
    "repetition_penalty": 1.0
}
project_id = "65870abf-b0eb-4dce-9b63-eeed50e3a3d0"
space_id = os.getenv("SPACE_ID")

try:
    modelallam = Model(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id,
        space_id=space_id
    )
    print("Model initialized successfully!")
except Exception as e:
    print(f"Error initializing the model: {e}")

# Define the input model
class UserInput(BaseModel):
    user_input: str



@app.post("/generate_poem/", response_class=HTMLResponse)
async def generate_poem(user_input: str = Form(...)):
    try:
        # Generate embedding for user input
        user_embedding = get_embedding(user_input)

        # Compute cosine similarity between user_embedding and all poem embeddings
        similarities = cosine_similarity(user_embedding, combined_embeddings)[0]

        # Get the indices of the top 10 most similar poems
        top_n = 10
        top_indices = similarities.argsort()[-top_n:][::-1]

        # Retrieve the top 10 similar poems
        retrieved_poems = []
        meter_count = {}
        for rank, idx in enumerate(top_indices, start=1):
            poem = full_poem_df.iloc[idx]['البيت']
            similarity_score = similarities[idx]
            metadata = {}
            for col in metadata_columns:
                if col in full_poem_df.columns:
                    metadata[col] = full_poem_df.iloc[idx][col]
                else:
                    metadata[col] = "Unknown"
            meter = metadata.get('البحر', 'Unknown')
            if meter != 'Unknown':
                meter_count[meter] = meter_count.get(meter, 0) + 1
            retrieved_poems.append((rank, idx, similarity_score, poem, metadata))

        # Determine the most common meter among the top 10 poems
        most_common_meter = max(meter_count, key=meter_count.get)

        # Retrieve the top 5 poems that match the common meter and prepare retry limits
        common_meter_rules = meter_rules.get(most_common_meter)
        top_poems = [poem for poem in retrieved_poems if poem[4]['البحر'] == most_common_meter][:5]
        max_retries = len(top_poems)

        def generate_prompt(user_input, poem):
            """
            Generates the prompt using a provided poem as an example.
            """
            formatted_question = f"""
            <s> [INST]
            أنت نموذج لغوي ذكي، مهمتك هي كتابة بيت شعري جديد باستخدام الوزن: "{common_meter_rules['description']}" ({most_common_meter}).
            سيتم تقديم مثال من الشعراء باستخدام نفس الوزن. يجب أن تعبر عن موضوع أو مشاعر تتعلق بـ "{user_input}" وأن تتبع قواعد هذا البحر الشعري بدقة.

            ### مثال من قصيدة باستخدام {most_common_meter}:
            "{poem[3]}"
               الشاعر: {poem[4]['الشاعر']}, العصر: {poem[4]['العصر']}
               الوزن: {common_meter_rules['pattern']}

            ### قواعد البحر الشعري ({most_common_meter}):
            - التفعيلات: {common_meter_rules['pattern']}
            - التقطيع العروضي: {common_meter_rules['prosody']}
            - مثال: "{common_meter_rules['example']}"

            ### الآن، اكتب بيتًا شعريًا يعبر عن "{user_input}"، مع الالتزام بقواعد {most_common_meter}.
            [/INST]
            </s>
            """
            return formatted_question

        def get_user_retry_input(attempt_number):
            """
            Prompt user to retry with a different example.
            """
            if attempt_number >= max_retries:
                return False

            user_response = input("هل ترغب في إعادة المحاولة باستخدام مثال مختلف؟ (نعم/لا): ").strip().lower()
            return user_response in ['نعم', 'yes']

        def generate_poem(user_input):
            """
            Generates a poem based on user input with retry capability.
            """
            for attempt in range(max_retries):
                current_poem = top_poems[attempt]
                prompt = generate_prompt(user_input, current_poem)

                # Generate the response using the model
                generated_response = modelallam.generate_text(prompt=prompt, guardrails=False)

                # Display generated poem
                print(f"\nالقصيدة المولدة (المحاولة {attempt + 1}):\n{generated_response}\n")

                # Ask the user if they want to retry with a different example
                if attempt < max_retries - 1:
                    retry = get_user_retry_input(attempt + 1)
                    if retry:
                        print("جاري إعادة المحاولة باستخدام مثال مختلف...\n")
                        continue
                    else:
                        print("تم إنهاء العملية.")
                        break
            else:
                print("لم يتم توليد قصيدة جديدة بعد المحاولات المتاحة.")

        # Example usage
        if __name__ == "__main__":
            generate_poem(user_input)

        # Retrieve the top 3 poems that follow the most common meter
        top_3_poems = [poem for poem in retrieved_poems if poem[4]['البحر'] == most_common_meter][:3]

        # Prepare the detailed prompt, including the top 3 poems as examples
        formattedQuestion = f"""
        <s> [INST]
        أنت نموذج لغوي ذكي، مهمتك هي كتابة بيت شعري جديد باستخدام الوزن: "{common_meter_rules['description']}" ({most_common_meter}). سيتم تقديم ثلاثة أمثلة من الشعراء باستخدام نفس الوزن. يجب أن تعبر عن موضوع أو مشاعر تتعلق بـ "{user_input}" وأن تتبع قواعد هذا البحر الشعري بدقة.

        ### أمثلة من أفضل ثلاث قصائد باستخدام {most_common_meter}:

        2. "{top_3_poems[0][3]}"

           الشاعر: {top_3_poems[0][4]['الشاعر']}, العصر: {top_3_poems[0][4]['العصر']}
           الوزن: {common_meter_rules['pattern']}

        ### قواعد البحر الشعري ({most_common_meter}):

        - التفعيلات: {common_meter_rules['pattern']}
        - التقطيع العروضي: {common_meter_rules['prosody']}
        - مثال: "{common_meter_rules['example']}"

        ### الآن، اكتب بيتًا شعريًا يعبر عن "{user_input}"، مع الالتزام بقواعد {most_common_meter}.
        [/INST]
        </s>
        """

        # Define the prompt
        prompt = f"""{formattedQuestion}"""

        # Generate the response using the model (assuming a model with a similar function to generate_text exists)
        generated_response = modelallam.generate_text(prompt=prompt, guardrails=False)

        # Create a cleaned-up validation prompt
        validation_prompt = create_clean_validation_prompt(generated_response, most_common_meter)

        # Generate the validation response using the model
        validation_response = modelallam.generate_text(prompt=validation_prompt, guardrails=False)

        # Return the results as HTML
        poetry_lines = generated_response.split('\n')
        poetry_lines_val = validation_response.split('\n')

        return f"""
        <div class="bg-white shadow-md rounded-lg overflow-hidden">
            <div class="border-r-4 border-indigo-500">
                <div class="p-6">
                    <div class="flex items-center gap-3 mb-6">
                        <i class="fas fa-scroll text-indigo-500 text-xl"></i>
                        <h3 class="text-xl font-bold text-gray-800">القصيدة المولدة</h3>
                    </div>
                    <div class="space-y-4">
                        <div class="poetry-lines space-y-3 text-center">
                            {''.join(f'<p class="text-lg text-gray-700 hover:text-indigo-600 transition-colors">{line}</p>' for line in poetry_lines if line.strip())}
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="bg-white shadow-md rounded-lg overflow-hidden mt-4">
            <div class="border-r-4 border-green-500">
                <div class="p-6">
                    <div class="flex items-center gap-3 mb-4">
                        <i class="fas fa-check-circle text-green-500 text-xl"></i>
                        <h3 class="text-xl font-bold text-gray-800">نتيجة المراجعة</h3>
                    </div>
                    <div class="poetry-lines space-y-3">
                            {''.join(f'<p class="text-lg text-gray-700 hover:text-indigo-600 transition-colors">{line}</p>' for line in poetry_lines_val if line.strip())}
                        </div>
                </div>
            </div>
        </div>

        <div class="mt-4">
            <button onclick="copyPoetry()" class="w-full bg-gray-100 text-gray-700 py-2 px-4 rounded-md hover:bg-gray-200 transition-colors duration-200 flex items-center justify-center gap-2">
                <i class="fas fa-copy"></i>
                نسخ القصيدة
            </button>
        </div>
        """
    except Exception as e:
        return f"""
        <h2 class="text-2xl font-bold text-red-700 mb-4">حدث خطأ أثناء معالجة الطلب:</h2>
        <p class="text-gray-700">{str(e)}</p>
        """
        
    




# =======================================================================
# ===============================Model 2=================================
# =======================================================================

@app.get("/model2", response_class=HTMLResponse)
async def read_model2():
    with open("./pages/model2.html", "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())
    

# Function to retrieve top 3 poems by the poet's common meter
def top_poems_by_common_meter(df, poet_name):
    poet_df = df[df["الشاعر"] == poet_name]

    # Check if the poet has poems available
    if poet_df.empty:
        return None, None

    # Find the most common meter for this poet
    most_common_meter = poet_df["البحر"].mode()[0]

    # Filter poems to those with the common meter and get the top 3
    top_poems = poet_df[poet_df["البحر"] == most_common_meter].head(3)

    return top_poems[["الشاعر", "البحر", "البيت"]], most_common_meter

# Define the input model for the API
class PoemRequest(BaseModel):
    poet_name: str
    user_topic: str

class EraRequest(BaseModel):
    era: str

class PoetRequest(BaseModel):
    poet_name: str

@app.get("/eras")
async def get_eras():
    eras = full_poem_df["العصر"].unique().tolist()
    return {"eras": eras}

@app.post("/poets")
async def get_poets(request: EraRequest):
    era = request.era
    poets = full_poem_df[full_poem_df["العصر"] == era]["الشاعر"].unique().tolist()
    return {"poets": poets}

@app.post("/poems")
async def get_poems(request: PoetRequest):
    poet_name = request.poet_name
    poems = full_poem_df[full_poem_df["الشاعر"] == poet_name]["البيت"].tolist()
    return {"poems": poems}

@app.post("/generate_poem_model/")
async def generate_poem_model(request: PoemRequest):
    poet_name = request.poet_name
    user_topic = request.user_topic

    # Get top poems and most common meter for the selected poet
    top_poems, most_common_meter = top_poems_by_common_meter(full_poem_df, poet_name)

    # Check if top poems are available
    if top_poems is not None and len(top_poems) >= 3:
        # Define meter rules
        common_meter_rules = meter_rules.get(most_common_meter)

        # Prepare the top 3 poems list for few-shot learning
        top_3_poems = top_poems.to_dict("records")[:3]

        # Construct the tailored prompt
        formatted_question = f"""
        <s> [INST]
        أنت نموذج لغوي ذكي، مهمتك هي كتابة بيت شعري جديد بأسلوب الشاعر: "{top_3_poems[0]['الشاعر']}"، باستخدام الوزن: "{common_meter_rules['description']}" ({most_common_meter}). سيتم تقديم ثلاثة أمثلة من الشاعر نفسه باستخدام نفس الوزن لتوضيح أسلوبه. يجب أن تعبر عن موضوع أو مشاعر تتعلق بـ "{user_topic}" وأن تتبع قواعد هذا البحر الشعري بدقة.

        ### أمثلة من أفضل ثلاث قصائد للشاعر باستخدام {most_common_meter}:

        1. "{top_3_poems[0]['البيت']}"

        2. "{top_3_poems[1]['البيت']}"

        3. "{top_3_poems[2]['البيت']}"

        ### قواعد البحر الشعري ({most_common_meter}):

        - التفعيلات: {common_meter_rules['pattern']}
        - التقطيع العروضي: {common_meter_rules['prosody']}
        - مثال: "{common_meter_rules['example']}"

        ### الآن، اكتب بيتًا شعريًا جديدًا يعبر عن "{user_topic}"، مع الالتزام بقواعد {most_common_meter}.
        [/INST]
        </s>
        """

        # Generate the response using the model
        generated_response_model = modelallam.generate_text(prompt=formatted_question, guardrails=False)

        # Return the AI-generated poem
        return {"generated_poem": generated_response_model}
    else:
        raise HTTPException(status_code=404, detail="Not enough poems available to create a prompt.")



# =======================================================================
# ===============================Model 3=================================
# =======================================================================

@app.get("/model3", response_class=HTMLResponse)
async def read_model3():
    with open("./pages/model3.html", "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())

# Define the data model for the poem data
class PoemData(BaseModel):
    sea: str
    verse: str
    poet: str
    dialect: str

    @validator('sea', 'verse', 'poet', 'dialect')
    def validate_arabic_characters(cls, value):
        arabic_pattern = re.compile(r'^[\u0600-\u06FF\s]+$')
        if not arabic_pattern.match(value):
            raise ValueError('يجب أن يحتوي الإدخال فقط على أحرف عربية ومسافات.')
        return value

# Initialize the CSV file if it doesn't exist
CSV_FILE_PATH = 'بيانات_القصائد_العربية.csv'
if not os.path.exists(CSV_FILE_PATH):
    pd.DataFrame(columns=['sea', 'verse', 'poet', 'dialect']).to_csv(CSV_FILE_PATH, index=False, encoding='utf-8-sig')

@app.post("/add_poem/")
async def add_poem(request: Request):
    form_data = await request.form()
    poem_data = PoemData(
        sea=form_data['sea'],
        verse=form_data['verse'],
        poet=form_data['poet'],
        dialect=form_data['dialect']
    )
    
    # Load existing data
    df = pd.read_csv(CSV_FILE_PATH, encoding='utf-8-sig')
    
    # Append new data
    new_data = pd.DataFrame([poem_data.dict()])
    df = pd.concat([df, new_data], ignore_index=True)
    
    # Save the updated data back to the CSV file
    df.to_csv(CSV_FILE_PATH, index=False, encoding='utf-8-sig')
    
    return {"message": "تمت إضافة القصيدة بنجاح."}

@app.get("/export_csv/")
def export_csv():
    # Load existing data
    df = pd.read_csv(CSV_FILE_PATH, encoding='utf-8-sig')
    
    # Convert DataFrame to CSV
    csv_data = df.to_csv(index=False, encoding='utf-8-sig')
    
    # Return the CSV data as a response
    return Response(content=csv_data, media_type="text/csv")

@app.get("/view_data/")
def view_data():
    # Load existing data
    df = pd.read_csv(CSV_FILE_PATH, encoding='utf-8-sig')
    
    # Return the data as a list of dictionaries
    return df.to_dict(orient='records')
