import streamlit as st
import cv2
import mediapipe as mp
import math
import numpy as np
import os
import random
import json
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pillow_heif import register_heif_opener
import google.generativeai as genai

# Mediapipeの安定インポート
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# iPhoneのHEIC形式通訳ライブラリ
register_heif_opener()

# ==========================================
# 🌟 ビジュアル設定 ＆ CSS
# ==========================================
st.set_page_config(layout="centered", page_title="AI姿勢分析システム")

st.markdown("""
<style>
    div[data-testid="stImage"] {
        display: flex;
        justify-content: center;
        margin-top: 15px;
        margin-bottom: -15px;
    }
    .custom-title {
        font-size: 24px !important;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
        color: #1F2937;
        letter-spacing: 1px;
    }
    @media (max-width: 600px) {
        .custom-title {
            font-size: 18px !important;
            letter-spacing: 0px !important;
            margin-bottom: 15px;
        }
    }
    .stFileUploader {
        padding-top: 0px;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# 公式ロゴの読み込み
logo_full_width = "PI＆ZERO100_ロゴ.jpg"
logo_half_width = "PI&ZERO100_ロゴ.jpg"

if os.path.exists(logo_full_width):
    st.image(logo_full_width, width=220)
elif os.path.exists(logo_half_width):
    st.image(logo_half_width, width=220)

st.markdown('<div class="custom-title">✨ AI 姿勢・骨盤総合分析システム ✨</div>', unsafe_allow_html=True)

# ==========================================
# 🧠 進化機能①：AI自己学習モジュール（ローカル保存）
# ==========================================
LEARNING_FILE = "ai_learning_data.json"

def load_learning_data():
    if os.path.exists(LEARNING_FILE):
        with open(LEARNING_FILE, "r") as f:
            return json.load(f)
    return {"count": 0, "offsets": {"e_x":0, "e_y":0, "s_x":0, "s_y":0, "h_x":0, "h_y":0, "k_x":0, "k_y":0, "a_x":0, "a_y":0}}

def save_learning_data(data):
    with open(LEARNING_FILE, "w") as f:
        json.dump(data, f)

def round_to_5(x):
    return 5 * round(x / 5)

learned_data = load_learning_data()
offsets = learned_data["offsets"]
ai_level = learned_data["count"]

# ==========================================
# 🔐 【最重要セキュリティ】Gemini APIキーの安全な読み込み
# ==========================================
gemini_ready = False
MY_GEMINI_API_KEY = ""

# 1. 優先して「Streamlitのシークレット（ネット上の見えない金庫）」から読み込む
try:
    MY_GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    # 2. 金庫になければ、ここの直書きを読み込む（※パソコンでテストする時だけここに入力します！）
    MY_GEMINI_API_KEY = "" # テスト時はこのダブルクォーテーションの間にキーを貼ります

# ▼ ここの条件をシンプルに「空っぽ（""）じゃなければ起動する」に変えます
if MY_GEMINI_API_KEY and MY_GEMINI_API_KEY != "":
    try:
        genai.configure(api_key=MY_GEMINI_API_KEY)
        gemini_ready = True
    except:
        pass

# サイドバー
with st.sidebar:
    st.write("### 📈 AI自己学習ステータス")
    st.write(f"現在のAIレベル: **Lv.{ai_level}**")
    st.caption("（微調整を学習した回数）")
    if gemini_ready:
        st.success("🌌 Gemini セキュア接続完了")
    else:
        st.error("Geminiと未接続です。テスト環境の場合はコード内にAPIキーを設定するか、クラウド環境の場合はSecretsにキーを設定してください。")

# AI検出器の準備
@st.cache_resource
def load_detector():
    model_path = "pose_landmarker_full.task"
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False
    )
    return vision.PoseLandmarker.create_from_options(options)

detector = load_detector()

def put_japanese_text(img, text, position, font_size, color, stroke_color=(0, 0, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("meiryo.ttc", font_size)
    except:
        font = ImageFont.load_default()
    b, g, r = color
    sb, sg, sr = stroke_color
    stroke_width = max(2, int(font_size * 0.12))
    draw.text(position, text, font=font, fill=(r, g, b), stroke_width=stroke_width, stroke_fill=(sr, sg, sb))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 患者様の基本情報
st.write("### 👤 患者様の基本情報")
col_g, col_a = st.columns(2)
with col_g:
    user_gender = st.radio("性別を選択", ["女性", "男性"], horizontal=True)
with col_a:
    user_age = st.number_input("実年齢", min_value=0, max_value=120, value=40, step=1)

# 画像アップロード
st.write("### 📷 写真のアップロード")
img_file = st.file_uploader("", type=['jpg', 'jpeg', 'png', 'heic', 'HEIC'])

if img_file is not None:
    image_raw = Image.open(img_file)
    image_pil = ImageOps.exif_transpose(image_raw)
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    detection_result = detector.detect(mp_image)
    
    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]
        
        is_right_facing = landmarks[0].x > landmarks[8].x
        if is_right_facing:
            ear_idx, sh_idx, hip_idx, knee_idx, ankle_idx = 7, 12, 24, 26, 28
            eye_idx, heel_idx = 5, 32
        else:
            ear_idx, sh_idx, hip_idx, knee_idx, ankle_idx = 8, 11, 23, 25, 27
            eye_idx, heel_idx = 2, 31
        
        nose_x, nose_y = int(landmarks[0].x * w), int(landmarks[0].y * h)
        eye_x, eye_y = int(landmarks[eye_idx].x * w), int(landmarks[eye_idx].y * h)
        sh_x, sh_y = int(landmarks[sh_idx].x * w), int(landmarks[sh_idx].y * h)
        hip_x, hip_y = int(landmarks[hip_idx].x * w), int(landmarks[hip_idx].y * h)
        knee_x, knee_y = int(landmarks[knee_idx].x * w), int(landmarks[knee_idx].y * h)
        
        if is_right_facing:
            dx = nose_x - eye_x
            estimated_ear_x = nose_x - int(dx * 3.2)
        else:
            dx = eye_x - nose_x
            estimated_ear_x = nose_x + int(dx * 3.2)
        dy = nose_y - eye_y
        estimated_ear_y = eye_y + int(dy * 1.3)
        
        ai_ear_x = int(landmarks[ear_idx].x * w)
        ai_ear_y = int(landmarks[ear_idx].y * h)
        ear_x = int(ai_ear_x * 0.2 + estimated_ear_x * 0.8)
        ear_y = int(ai_ear_y * 0.2 + estimated_ear_y * 0.8)

        ankle_x = int(landmarks[ankle_idx].x * w)
        heel_y = int(landmarks[heel_idx].y * h)
        ankle_y = heel_y - int(h * 0.012) 

        report_spot = st.container()
        image_spot = st.container()
        
        # --- 術者用スライダー ---
        st.markdown("---")
        st.write("### 🛠️ 術者用・各関節位置の微調整")
        if ai_level > 0:
            st.caption(f"🧠 **AIからのメッセージ**: 過去 {ai_level} 回の学習データから、あなたに合わせた補正値を自動でセットしました！")
        else:
            st.caption("ブルーの縦線目盛り（5ピクセル刻み）につまみが完全に同期して重なります。")
        
        tab_ear, tab_sh, tab_hip, tab_knee, tab_ankle = st.tabs([
            "🔴 耳（耳垂）", "🟢 肩（肩峰）", "🟡 大転子（腰）", "🟣 ひざ（関節裂隙）", "🌸 足首（外くるぶし）"
        ])
        
        adj_options = list(range(-50, 51, 5))
        tick_html = '<div style="display:flex; justify-content:space-between; padding:0 10px; margin-top:-22px; margin-bottom:15px; color:#0066cc; font-weight:bold; font-size:11px; letter-spacing:-1px; user-select:none;">' + ''.join(['<div>|</div>' for _ in adj_options]) + '</div>'
        
        with tab_ear:
            e_x = st.select_slider("耳垂：左右", options=adj_options, value=round_to_5(offsets["e_x"]), key="e_x")
            st.markdown(tick_html, unsafe_allow_html=True)
            e_y = st.select_slider("耳垂：上下", options=adj_options, value=round_to_5(offsets["e_y"]), key="e_y")
            st.markdown(tick_html, unsafe_allow_html=True)
        with tab_sh:
            s_x = st.select_slider("肩峰：左右", options=adj_options, value=round_to_5(offsets["s_x"]), key="s_x")
            st.markdown(tick_html, unsafe_allow_html=True)
            s_y = st.select_slider("肩峰：上下", options=adj_options, value=round_to_5(offsets["s_y"]), key="s_y")
            st.markdown(tick_html, unsafe_allow_html=True)
        with tab_hip:
            h_x = st.select_slider("大転子：左右", options=adj_options, value=round_to_5(offsets["h_x"]), key="h_x")
            st.markdown(tick_html, unsafe_allow_html=True)
            h_y = st.select_slider("大転子：上下", options=adj_options, value=round_to_5(offsets["h_y"]), key="h_y")
            st.markdown(tick_html, unsafe_allow_html=True)
        with tab_knee:
            k_x = st.select_slider("ひざ：左右", options=adj_options, value=round_to_5(offsets["k_x"]), key="k_x")
            st.markdown(tick_html, unsafe_allow_html=True)
            k_y = st.select_slider("ひざ：上下", options=adj_options, value=round_to_5(offsets["k_y"]), key="k_y")
            st.markdown(tick_html, unsafe_allow_html=True)
        with tab_ankle:
            a_x = st.select_slider("足首：左右", options=adj_options, value=round_to_5(offsets["a_x"]), key="a_x")
            st.markdown(tick_html, unsafe_allow_html=True)
            a_y = st.select_slider("足首：上下", options=adj_options, value=round_to_5(offsets["a_y"]), key="a_y")
            st.markdown(tick_html, unsafe_allow_html=True)

        if st.button("🎓 今の微調整のクセをAIに学習させる"):
            new_c = ai_level + 1
            def update_avg(old, new, count):
                return ((old * (count - 1)) + new) / count
            
            new_offsets = {
                "e_x": update_avg(offsets["e_x"], e_x, new_c), "e_y": update_avg(offsets["e_y"], e_y, new_c),
                "s_x": update_avg(offsets["s_x"], s_x, new_c), "s_y": update_avg(offsets["s_y"], s_y, new_c),
                "h_x": update_avg(offsets["h_x"], h_x, new_c), "h_y": update_avg(offsets["h_y"], h_y, new_c),
                "k_x": update_avg(offsets["k_x"], k_x, new_c), "k_y": update_avg(offsets["k_y"], k_y, new_c),
                "a_x": update_avg(offsets["a_x"], a_x, new_c), "a_y": update_avg(offsets["a_y"], a_y, new_c),
            }
            save_learning_data({"count": new_c, "offsets": new_offsets})
            st.success(f"学習完了！AIが Lv.{new_c} にレベルアップしました！次からこの癖を自動で補正します。")

        ear_x += e_x; ear_y -= e_y
        sh_x += s_x; sh_y -= s_y
        hip_x += h_x; hip_y -= h_y
        knee_x += k_x; knee_y -= k_y
        ankle_x += a_x; ankle_y -= a_y

        if is_right_facing:
            neck_dx = ear_x - sh_x; trunk_dx = sh_x - hip_x; thigh_dx = hip_x - knee_x; sway_dx = hip_x - ankle_x 
        else:
            neck_dx = sh_x - ear_x; trunk_dx = hip_x - sh_x; thigh_dx = knee_x - hip_x; sway_dx = ankle_x - hip_x 
        neck_dy = sh_y - ear_y; trunk_dy = hip_y - sh_y; thigh_dy = knee_y - hip_y; sway_dy = ankle_y - hip_y
        
        neck_angle = math.degrees(math.atan2(neck_dx, neck_dy))
        trunk_angle = math.degrees(math.atan2(trunk_dx, trunk_dy))
        thigh_angle = math.degrees(math.atan2(thigh_dx, thigh_dy))
        pelvis_angle = trunk_angle - thigh_angle
        sway_angle = math.degrees(math.atan2(sway_dx, sway_dy))

        neck_loss = min(35, abs(neck_angle) * 1.5)
        pelvis_loss = min(35, abs(pelvis_angle) * 3.0)
        sway_loss = min(30, abs(sway_angle) * 3.0)
        score = max(0, int(100 - (neck_loss + pelvis_loss + sway_loss)))

        if score >= 90:
            posture_age = max(18, user_age - 3)
            age_comment = "実年齢より若々しく美しい、理想的な骨格状態です✨"
        elif score >= 75:
            posture_age = user_age + 2
            age_comment = "大きな問題はありませんが、適度なメンテナンスがおすすめな状態です。"
        elif score >= 60:
            posture_age = user_age + 5
            age_comment = "筋肉の保持力が低下し、歪みが定着し始めている傾向があります。"
        else:
            posture_age = user_age + 10
            age_comment = "骨格への負担が蓄積しており、根本的なケアが急務な状態です🚨"

        random.seed(int(abs(neck_angle*100) + abs(pelvis_angle*100) + abs(sway_angle*100)))

        def get_random_risks(risk_list, num=2):
            selected = random.sample(risk_list, min(num, len(risk_list)))
            return "\n".join([f"* **{k}**\n  {v}" for k, v in selected])

        sway_fwd_risks = [
            ("慢性的な冷え・むくみ", "太もも前とふくらはぎだけで全体重を支えるため、下肢の血流やリンパの循環が低下します。"),
            ("足底腱膜炎・外反母趾の悪化", "足裏への不自然な荷重が続き、激痛や関節の変形を誘発します。"),
            ("前ももの過剰な張り・太さ", "常に前傾姿勢をブレーキするため、太もも前側の筋肉だけが過剰に発達してしまいます。"),
            ("巻き爪・タコ・ウオノメ", "足先の特定のポイントに体重が集中し続けることで、足先の皮膚や爪に異常をきたします。")
        ]
        sway_bwd_risks = [
            ("ギックリ腰体質の固定化", "日常の些細な動作（靴下を履く、物を拾うなど）で何度も腰を痛めるようになります。"),
            ("慢性背部痛の定着", "後ろに倒れないよう背中の筋肉が常にちぎれそうなほど緊張し続けます。"),
            ("かかとの痛み・アキレス腱炎", "かかとに体重が集中しすぎるため、歩行時や起床時に足首付近に激痛が走るリスクが高まります。"),
            ("首・肩の過緊張", "重心が後ろにある分、頭を前に出してバランスを取ろうとするため、首肩の負担が倍増します。")
        ]
        neck_mid_risks = [
            ("頻繁な寝違え", "首元の筋肉が常に過緊張を起こしているため、朝起きるたびに首が回らなくなる「寝違え」を繰り返しやすくなります。"),
            ("四十肩・五十肩", "頭の重みで首肩の血流が途絶え、関節が線維化（ガチガチにロック）します。"),
            ("慢性緊張性頭痛", "首元の筋肉の酸欠状態が定着し、薬が手放せない頭痛につながります。"),
            ("睡眠の質の低下", "首まわりの筋肉が寝ている間も緊張し続け、朝起きても疲れが取れない体質になります。"),
            ("眼精疲労と視力低下", "首から後頭部にかけての筋肉が固まり、目に向かう血流が阻害されます。")
        ]
        neck_sev_risks = [
            ("頻繁な寝違え（重症化）", "ストレートネックにより首の遊びがゼロになっているため、ちょっとした寝返りで首を激しく痛めるようになります。"),
            ("頸椎ヘルニア（手のしびれ）", "頸椎のクッションが完全に潰され、神経を激しく圧迫します。"),
            ("自律神経失調（不眠・めまい）", "首まわりの重要な自律神経ルートが圧迫され、慢性疲労を引き起こします。"),
            ("逆流性食道炎", "胸郭が内側に潰れ、内臓（胃など）を上から常に圧迫し続けます。"),
            ("顔のたるみ・二重あご", "首が前に出ることで顔まわりの筋膜が下に引っ張られ、実年齢より老けた印象が定着します。")
        ]
        pelvis_fwd_risks = [
            ("脊柱管狭窄症・歩行困難", "腰椎が過剰に反るため骨同士が衝突して変形し、神経の通り道を狭くします。"),
            ("坐骨神経痛（足の激痛・しびれ）", "お尻から足へ走る太い坐骨神経が骨盤の歪みで引き潰されます。"),
            ("ぽっこりお腹（内臓下垂）", "インナーマッスルが使えず、支えを失った内臓が下へ垂れ下がります。"),
            ("股関節の詰まり・痛み", "骨盤が前に倒れることで股関節の可動域が制限され、歩くたびに違和感や痛みが生じます。"),
            ("O脚・X脚の悪化", "股関節が内側にねじれやすくなり、膝への負担が増えることでO脚やX脚の変形が進行します。")
        ]
        pelvis_bwd_risks = [
            ("魔のギックリ腰の連発", "腰の自然なカーブが失われているため、顔を洗う・荷物を持つなどの些細な前屈み動作でギックリ腰を連発します。"),
            ("腰椎椎間板ヘルニア", "骨盤が後ろに寝ることで、椎間板が後ろへ飛び出す強い圧力がかかります。"),
            ("変形性膝関節症（歩行困難）", "骨盤で吸収できない歩行の衝撃が膝へ直撃し、軟骨をすり減らします。"),
            ("お尻のたるみ・扁平化", "臀部の筋肉が全く使われない状態が続くため、お尻のお肉が重力で下垂します。"),
            ("O脚（ガニ股）の進行", "骨盤が後傾することで太ももの骨が外側に開きやすくなり、膝の間が開くO脚が進行します。")
        ]

        if user_gender == "女性":
            pelvis_fwd_risks.append(("重い生理痛・PMSの悪化", "骨盤が前傾して内臓が下垂し、子宮や卵巣が圧迫されて骨盤内の血流が著しく滞ります。"))
            pelvis_fwd_risks.append(("重い生理痛・PMSの悪化", "骨盤が前傾して内臓が下垂し、子宮や卵巣が圧迫されて骨盤内の血流が著しく滞ります。")) 
            pelvis_bwd_risks.append(("重い生理痛・PMSの悪化", "骨盤周りの筋肉が硬直して血流が滞り、冷えが強まることで婦人科系のトラブルを誘発します。"))
            pelvis_bwd_risks.append(("重い生理痛・PMSの悪化", "骨盤周りの筋肉が硬直して血流が滞り、冷えが強まることで婦人科系のトラブルを誘発します。"))

        if -2 <= sway_angle <= 2:
            sway_status, sway_color = "重心位置：理想的", "success"
            sway_info = "くるぶしの真上に股関節が乗っており、足裏全体でバランス良く体重を支えられています。"
            sway_risk = "🟢 **リスクなし**\n関節や足底への部分的な過負荷は見られません。"
        elif sway_angle > 2:
            sway_status, sway_color = "前方シフト (スウェイバック)", "error"
            sway_info = "くるぶしよりも股関節が前方にスライドしています。太もも前とお腹の力に頼って立つため、腰椎への負担が非常に大きいです。"
            sway_risk = f"🚨 **放置した場合の将来リスク**\n{get_random_risks(sway_fwd_risks, 2)}"
        else:
            sway_status, sway_color = "後方シフト (かかと重心)", "warning"
            sway_info = "くるぶしよりも重心が後ろ（かかと側）に逃げています。バランスをとるために背中や首が前に倒れやすくなります。"
            sway_risk = f"🚨 **放置した場合の将来リスク**\n{get_random_risks(sway_bwd_risks, 2)}"

        if neck_angle < 5: 
            neck_status, neck_color = "理想的 (ストレート)", "success"
            neck_risk = "🟢 **リスクなし**\n頸椎椎間板への異常な変形ストレスはありません。"
        elif neck_angle < 15: 
            neck_status, neck_color = "やや猫背気味 (巻き肩)", "warning"
            neck_risk = f"⚠️ **放置した場合の将来リスク**\n{get_random_risks(neck_mid_risks, 2)}"
        else: 
            neck_status, neck_color = "強い猫背 (ストレートネック)", "error"
            neck_risk = f"🚨 **放置した場合の将来リスク**\n{get_random_risks(neck_sev_risks, 2)}"
            
        if -3 <= pelvis_angle <= 3: 
            pelvis_status, pelvis_color = "理想的な骨盤角度", "success"
            pelvis_risk = "🟢 **リスクなし**\n腰椎のカーブが正常で、神経圧迫のリスクは極めて低いです。"
        elif pelvis_angle > 3: 
            pelvis_status, pelvis_color = "骨盤前傾 (反り腰)", "error"
            pelvis_risk = f"🚨 **放置した場合の将来リスク**\n{get_random_risks(pelvis_fwd_risks, 2)}"
        else: 
            pelvis_status, pelvis_color = "骨盤後傾 (丸まり)", "warning"
            pelvis_risk = f"🚨 **放置した場合の将来リスク**\n{get_random_risks(pelvis_bwd_risks, 2)}"

        neck_ideal_patterns = [
            {"k": "少陽胆経・陽明大腸経", "d": "良好なバランスです。疲労を溜めないよう、気血の巡りを円滑に保つメンテナンスを行います。", "t": "風池（ふうち）、合谷（ごうこく）"},
            {"k": "陽明胃経・手少陽三焦経", "d": "首の軸は綺麗です。ストレスによる自律神経の乱れを防ぎ、上半身の軽さを維持します。", "t": "足三里（あしさんり）、外関（がいかん）"},
            {"k": "手厥陰心包経・太陰肺経", "d": "姿勢は保たれています。呼吸を深く保つことで、胸まわりの筋肉が固まるのを未然に防ぎます。", "t": "内関（ないかん）、列缺（れっけつ）"}
        ]
        neck_mild_patterns = [
            {"k": "太陰肺経・少陽胆経", "d": "巻き肩により胸の「肺経」が縮んでいます。胸郭を広げ、肩峰を通る「胆経」の気滞を抜いて姿勢を戻します。", "t": "中府（ちゅうふ）、肩井（けんせい）"},
            {"k": "陽明大腸経・太陽小腸経", "d": "腕の使いすぎで肩甲骨が外に引っ張られています。腕から肩につながる経絡を緩め、背中を楽にします。", "t": "手三里（てさんり）、腕骨（わんこつ）"},
            {"k": "陽明胃経・任脈", "d": "デスクワーク等で体の前面が縮こまっています。鎖骨周りやお腹の経絡を開き、自然に胸が張れる状態を作ります。", "t": "欠盆（けつぼん）、膻中（だんちゅう）"}
        ]
        neck_severe_patterns = [
            {"k": "太陰肺経・太陽小腸経", "d": "肩が著しく内巻きになり背部がガチガチです。肺経を緩めて呼吸を深くし、背面の「小腸経」の陽気を高めます。", "t": "中府（ちゅうふ）、後谿（こうけい）"},
            {"k": "太陽膀胱経・督脈", "d": "頭の重みで首の根元（膀胱経）が極度に過緊張しています。背骨のラインを緩め、自律神経の圧迫を解放します。", "t": "天柱（てんちゅう）、大椎（だいつい）"},
            {"k": "手少陽三焦経・少陽胆経", "d": "ストレートネックにより頭への血流が滞っています。耳の裏から首すじの経絡を流し、慢性頭痛や眼精疲労を防ぎます。", "t": "翳風（えいふう）、風池（ふうち）"}
        ]
        pelvis_ideal_patterns = [
            {"k": "任脈・陽明胃経", "d": "理想的な骨盤角度です。身体の前面を走る任脈（丹田）を安定させ、今の良い状態をキープします。", "t": "気海（きかい）、足三里（あしさんり）"},
            {"k": "太陰脾経・厥陰肝経", "d": "骨盤のバランスは良好です。下半身の血流を保ち、日常の疲労を翌日に持ち越さないケアを行います。", "t": "三陰交（さんいんこう）、太衝（たいしょう）"},
            {"k": "少陰腎経・太陽膀胱経", "d": "腰椎への負担が少ない状態です。足裏から腰までの軸をしっかり保つことで、歩行時の安定感を高めます。", "t": "太渓（たいけい）、委中（いちゅう）"}
        ]
        pelvis_fwd_patterns = [
            {"k": "太陽膀胱経・陽明胃経", "d": "反り腰で過緊張している腰（膀胱経）を緩め、骨盤を前に引っ張っている前もも（胃経）の突っ張りを引き剥がします。", "t": "腎兪（じんゆ）、梁丘（りょうきゅう）"},
            {"k": "少陽胆経・太陰脾経", "d": "骨盤の前傾によって股関節周辺が詰まっています。お尻の横（胆経）から内ももにかけての巡りを改善し、骨盤を立てます。", "t": "環跳（かんちょう）、血海（けっかい）"},
            {"k": "任脈・少陰腎経", "d": "前重心でお腹の力が抜けてしまっています。丹田（関元）に気血を集め、インナーマッスルで腰を守る力を復活させます。", "t": "関元（かんげん）、照海（しょうかい）"}
        ]
        pelvis_bwd_patterns = [
            {"k": "太陽膀胱経・少陽胆経", "d": "骨盤が後ろに寝ることで硬化した裏もも（膀胱経）とお尻の外側（胆経）を強力に緩め、正常な可動性を復活させます。", "t": "委中（いちゅう）、環跳（かんちょう）"},
            {"k": "厥陰肝経・少陰腎経", "d": "骨盤後傾により下半身の血流が滞り冷えやすくなっています。足元から気血を押し上げ、骨盤を立てる力を養います。", "t": "太衝（たいしょう）、復溜（ふくりゅう）"},
            {"k": "陽明胃経・太陰脾経", "d": "骨盤が寝ることで膝に過剰な負担がかかっています。すねから膝回りの経絡を流し、下半身の重だるさを抜いていきます。", "t": "豊隆（ほうりゅう）、陰陵泉（いんりょうせん）"}
        ]

        if neck_angle < 5: selected_neck = random.choice(neck_ideal_patterns)
        elif neck_angle < 15: selected_neck = random.choice(neck_mild_patterns)
        else: selected_neck = random.choice(neck_severe_patterns)

        if -3 <= pelvis_angle <= 3: selected_pelvis = random.choice(pelvis_ideal_patterns)
        elif pelvis_angle > 3: selected_pelvis = random.choice(pelvis_fwd_patterns)
        else: selected_pelvis = random.choice(pelvis_bwd_patterns)

        neck_tsubo_info = f"☯️ **主治経絡**：{selected_neck['k']}\n💡 **アプローチ**：{selected_neck['d']}\n📍 **推奨経穴**：{selected_neck['t']}"
        pelvis_tsubo_info = f"☯️ **主治経絡**：{selected_pelvis['k']}\n💡 **アプローチ**：{selected_pelvis['d']}\n📍 **推奨経穴**：{selected_pelvis['t']}"
        n_tsubo = selected_neck['t']
        p_tsubo = selected_pelvis['t']

        annotated_image = image.copy()
        cv2.line(annotated_image, (ankle_x, ankle_y), (ankle_x, 0), (255, 255, 255), 2)
        cv2.line(annotated_image, (sh_x, sh_y), (ear_x, ear_y), (255, 0, 0), 4)       
        cv2.line(annotated_image, (hip_x, hip_y), (sh_x, sh_y), (0, 255, 0), 4)       
        cv2.line(annotated_image, (knee_x, knee_y), (hip_x, hip_y), (0, 165, 255), 4) 
        cv2.line(annotated_image, (ankle_x, ankle_y), (knee_x, knee_y), (255, 105, 180), 4) 
        cv2.circle(annotated_image, (ear_x, ear_y), 12, (0, 0, 255), -1)
        cv2.circle(annotated_image, (sh_x, sh_y), 12, (0, 255, 0), -1)
        cv2.circle(annotated_image, (hip_x, hip_y), 12, (0, 255, 255), -1)
        cv2.circle(annotated_image, (knee_x, knee_y), 12, (255, 0, 255), -1)
        cv2.circle(annotated_image, (ankle_x, ankle_y), 12, (255, 105, 180), -1)
        
        font_size = max(32, int(w * 0.038))  
        line_height = int(font_size * 1.5)   
        margin_x = int(w * 0.03)             
        margin_y = int(h * 0.02)             

        annotated_image = put_japanese_text(annotated_image, f"猫背度: {neck_angle:.1f}度", (margin_x, margin_y), font_size, (0, 255, 255))
        annotated_image = put_japanese_text(annotated_image, f"骨盤歪み: {pelvis_angle:.1f}度", (margin_x, margin_y + line_height), font_size, (255, 255, 0))
        annotated_image = put_japanese_text(annotated_image, f"重心ズレ: {sway_angle:.1f}度", (margin_x, margin_y + line_height * 2), font_size, (255, 255, 255))

        with report_spot:
            st.subheader("📊 姿勢・骨盤総合分析レポート")
            col0, col1, col2, col3 = st.columns(4)
            with col0:
                st.metric(label="🏆 総合姿勢スコア", value=f"{score} 点", delta=f"想定骨格年齢: {posture_age}歳", delta_color="inverse")
                st.caption(f"{age_comment}")
            with col1:
                st.metric(label="① 首の傾き", value=f"{neck_angle:.1f} 度")
            with col2:
                st.metric(label="② 骨盤の傾き", value=f"{pelvis_angle:.1f} 度")
            with col3:
                st.metric(label="③ 重心シフト", value=f"{sway_angle:.1f} 度")
            st.markdown("---")
            
        with image_spot:
            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="AIゴールデンライン分析", use_container_width=True)

        with st.expander("🔍 西洋医学的分析・放置した場合の将来リスク（クリックで展開）"):
            st.write("### 🟥 【首・肩・上半身の状態】")
            st.write(f"**現在の状態**：{ '肩峰が本来の位置より前に巻き込まれ、大胸筋が硬く縮み、背中が引っ張られています。' if neck_angle >= 5 else '耳垂から肩峰にかけての前後バランスが良く、頭の重さを背骨全体で上手に支えられています。' }")
            st.info(neck_risk)
            st.write("---")
            st.write("### 🟨 【骨盤・下半身の状態】")
            if pelvis_angle > 3: st.write("**現在の状態**：大転子が前方へ傾き、太もも前や腰が過緊張し、お腹とお尻が使えていません。")
            elif pelvis_angle < -3: st.write("**現在の状態**：大転子が後方へ傾き、太もも裏が硬く縮み、骨盤を後ろに引っ張っています。")
            else: st.write("**現在の状態**：お腹とお尻の筋肉バランスが非常に良く、腰椎の負担が少ない理想的な状態です。")
            st.info(pelvis_risk)
            st.write("---")
            st.write("### ⬜ 【重心（足首〜股関節）】")
            st.write(f"**現在の状態**：{sway_info}")
            st.info(sway_risk)

        with st.expander("☯️ 東洋医学アプローチ（主治経絡・推奨経穴）（クリックで展開）"):
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("**【首・肩・上半身からのアプローチ】**")
                st.success(neck_tsubo_info)
            with col_t2:
                st.markdown("**【骨盤・下半身からのアプローチ】**")
                st.success(pelvis_tsubo_info)

        with st.expander("📱 患者様へLINE送信テキスト（クリックで展開）", expanded=False):
            st.caption("右上のコピーボタンを押すと、このまま院の公式LINE等から患者様へ送信できます。")
            line_text = f"""【AI姿勢・骨盤総合分析レポート】
本日もお疲れ様でした！お体の分析結果をお届けします。

━━━━━━━━━━━━━━
🏆 総合姿勢スコア：{score}点
👤 患者様の実年齢：{user_age}歳
🔥 算出した想定姿勢年齢：{posture_age}歳相当
（骨格状態：{age_comment}）
━━━━━━━━━━━━━━

■ 本日の分析結果
①首の傾き：{neck_angle:.1f}度（{neck_status}）
②骨盤の傾き：{pelvis_angle:.1f}度（{pelvis_status}）
③全体の重心：{sway_angle:.1f}度（{sway_status}）

■ セルフケア・おすすめのツボ
・上半身：{n_tsubo}
・下半身：{p_tsubo}

次回のご来院時に、このスコアと姿勢年齢がどう若返るか再検査しましょう！
一緒に理想の100点満点を目指して頑張りましょう。"""
            st.code(line_text, language="text")

      # ==========================================
        # 🧠 進化機能②：Geminiによる専用カルテ無限生成
        # ==========================================
        if gemini_ready:
            st.markdown("---")
            st.write("### 🌌 Gemini 専属AIアシスタント機能")
            if st.button("✨ この患者様だけの専用カウンセリングレポートを生成する"):
                with st.spinner('Geminiが膨大な医学データから専用のカルテを執筆中です...'):
                    prompt = f"""
                    あなたはトップレベルの治療家であり、東洋医学にも精通したプロフェッショナルです。
                    以下の患者データをもとに、患者様の心に寄り添い、かつ危機感と希望を与える「専用カウンセリングレポート」を3〜4つの短い段落で作成してください。
                    専門用語は避け、素人にも分かりやすく、優しく前向きなトーンで書いてください。「100歳相当」のような過度な煽りは厳禁です。
                    最後に、この状態に最適な東洋医学のツボを2つ提案し、その理由を簡潔に添えてください。

                    【患者データ】
                    ・年齢: {user_age}歳
                    ・性別: {user_gender}
                    ・姿勢スコア: {score}点/100点
                    ・首の傾き: {neck_angle:.1f}度（数値が大きいほどストレートネック・猫背）
                    ・骨盤の傾き: {pelvis_angle:.1f}度（プラスは反り腰、マイナスは丸まり）
                    ・重心ズレ: {sway_angle:.1f}度（プラスは前方、マイナスは後方）
                    """
                    try:
                        # 💡 先生の環境で使える最新の「Gemini 3」モデルを直接指名します！
                        model = genai.GenerativeModel('gemini-3-flash-preview')
                        response = model.generate_content(prompt)
                        st.success("✅ Geminiによる専用レポートが完成しました！")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"Gemini APIとの通信でエラーが発生しました。（エラー詳細: {e}）")
        else:
            st.info("💡 コード内に安全なAPIキーが設定されるか、クラウドのSecrets設定が完了すると、AIが毎回患者様ごとの『世界に一つだけの長文分析レポート』を自動生成する機能が解放されます！")

    else:
        st.error("人がうまく認識できませんでした。もう少し離れて全身が写るように撮影してください。")