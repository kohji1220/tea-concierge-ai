# クラウド環境(Streamlit Cloud)の古いSQLiteバージョン対策ハック
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import json
import os
import requests
import base64
import time
import hashlib
import streamlit as st
import chromadb

# ==========================================
# 1. 初期設定 & UI (CSS)
# ==========================================
st.set_page_config(page_title="AI Tea Concierge & Analyzer", page_icon="🍵", layout="centered")

# ダークモード/ライトモードの両方に適応するよう、背景色・文字色の強制指定を解除
st.markdown("""
<style>
    .stApp { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    h1 { font-weight: 300; letter-spacing: 2px; border-bottom: 1px solid #D5DBDB; padding-bottom: 10px; }
    .stChatInputContainer { border-radius: 20px !important; }
    /* サクラチェッカー用の強調表示 */
    .risk-high { color: #E74C3C; font-weight: bold; font-size: 1.2em; }
    .risk-medium { color: #F39C12; font-weight: bold; font-size: 1.2em; }
    .risk-low { color: #27AE60; font-weight: bold; font-size: 1.2em; }
</style>
""", unsafe_allow_html=True)

if "GEMINI_API_KEY" in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    API_KEY = "あなたのAPIキーをここに入力してください"

CHROMA_DB_PATH = "./tea_chroma_db"

# ==========================================
# 2. ベクトルデータベース (ChromaDB) エンジン
# ==========================================

def call_api_with_retry(url, headers, payload):
    """APIの制限や混雑時に、待機時間を延ばしながら自動で再試行するヘルパー関数"""
    delays = [1, 2, 4, 8, 16]
    for attempt in range(len(delays) + 1):
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=60)
            if res.status_code == 200:
                return res
            # 429(制限到達) または 503(サーバー高負荷) の場合のみリトライ
            elif res.status_code in [429, 503] and attempt < len(delays):
                time.sleep(delays[attempt])
                continue
            else:
                return res
        except requests.exceptions.RequestException as e:
            if attempt < len(delays):
                time.sleep(delays[attempt])
                continue
            raise e
    return None

def get_embedding(text):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}
    payload = {"model": "models/text-embedding-004", "content": {"parts": [{"text": text}]}}
    try:
        res = call_api_with_retry(url, headers, payload)
        if res and res.status_code == 200:
            return res.json()['embedding']['values']
        return None
    except Exception:
        return None

def generate_id(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

@st.cache_resource(show_spinner=False)
def init_chromadb():
    file_path = "web_tea_knowledge.jsonl"
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name="tea_knowledge_collection")
    
    if not os.path.exists(file_path):
        return collection

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            return collection

        existing_data = collection.get()
        existing_ids = set(existing_data['ids']) if existing_data and 'ids' in existing_data else set()

        new_docs = []
        new_ids = []
        for line in lines:
            data = json.loads(line)
            text_content = json.dumps(data, ensure_ascii=False)
            doc_id = generate_id(text_content)
            if doc_id not in existing_ids:
                new_docs.append(text_content)
                new_ids.append(doc_id)

        if new_docs:
            st.info(f"🍵 {len(new_docs)}件の新規データを検出。インデックスを構築中...")
            progress_bar = st.progress(0)
            for i, (doc, doc_id) in enumerate(zip(new_docs, new_ids)):
                vector = get_embedding(doc)
                if vector:
                    collection.add(
                        documents=[doc],
                        embeddings=[vector],
                        metadatas=[{"source": "web_tea_knowledge"}],
                        ids=[doc_id]
                    )
                time.sleep(0.05)
                progress_bar.progress((i + 1) / len(new_docs))
            progress_bar.empty()
        return collection
    except Exception as e:
        return client.get_or_create_collection(name="tea_knowledge_collection")

# ==========================================
# 3. アプリケーションの初期化
# ==========================================

with st.spinner("Initializing Vector Database..."):
    collection = init_chromadb()
    db_count = collection.count()

with st.sidebar:
    st.markdown("### 🍵 System Status")
    st.success(f"ChromaDB Active ({db_count} records)")
    st.success("Google Search Grounding: Ready")
    st.success("Multimodal Vision: Ready")
    st.markdown("---")

st.title("AI Tea Concierge & Analyzer")

# ==========================================
# 4. タブによる機能切り替え (StreamlitのUI機能)
# ==========================================

tab1, tab2 = st.tabs(["💬 コンシェルジュ (通常対話)", "🚨 査定チェッカー (怪しさ判定)"])

# ------------------------------------------
# タブ1: 通常のコンシェルジュチャット
# ------------------------------------------
with tab1:
    st.markdown("*Advanced Reasoning Engine for Tea Science & Market Economy*")
    
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
        
    uploaded_file = st.file_uploader(
        "茶葉や水色の画像をアップロード (任意)", 
        type=["png", "jpg", "jpeg", "webp"], 
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.api_history = []
        greeting = "いらっしゃいませ。最新鋭の推論エンジンと専用インデックスが稼働しております。どのようなご質問でもお聞かせください。"
        st.session_state.messages.append({"role": "assistant", "content": {"text": greeting}})

    for msg in st.session_state.messages:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤵"
        with st.chat_message(msg["role"], avatar=avatar):
            content = msg["content"]
            if "text" in content:
                st.markdown(content["text"])
            if "image_bytes" in content:
                st.image(content["image_bytes"], width=250)

    if prompt := st.chat_input("ご質問を入力してください..."):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
            if uploaded_file is not None:
                st.image(uploaded_file, width=250)

        user_parts = [{"text": prompt}]
        message_content = {"text": prompt}

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            base64_data = base64.b64encode(file_bytes).decode("utf-8")
            user_parts.append({"inlineData": {"mimeType": uploaded_file.type, "data": base64_data}})
            message_content["image_bytes"] = file_bytes
            st.session_state.uploader_key += 1

        st.session_state.messages.append({"role": "user", "content": message_content})
        
        with st.chat_message("assistant", avatar="🤵"):
            with st.spinner("Analyzing knowledge index and current web data..."):
                query_emb = get_embedding(prompt)
                relevant_knowledge = ""
                
                if query_emb and collection.count() > 0:
                    results = collection.query(query_embeddings=[query_emb], n_results=min(3, collection.count()))
                    if results['documents'] and len(results['documents']) > 0:
                        relevant_knowledge = "\n".join(results['documents'][0])

                dynamic_system_prompt = f"""
                あなたは世界トップクラスの知識を持つ「お茶の専属コンシェルジュ」です。
                以下の【関連する専門知識 (ChromaDB抽出)】を最優先の根拠とし、不足分はGoogle検索で補って回答してください。
                【関連する専門知識】\n{relevant_knowledge if relevant_knowledge else "なし"}
                """

                current_api_history = st.session_state.api_history.copy()
                current_api_history.append({"role": "user", "parts": user_parts})

                payload = {
                    "systemInstruction": {"parts": [{"text": dynamic_system_prompt}]},
                    "contents": current_api_history,
                    "tools": [{"google_search": {}}], # コンシェルジュは検索利用可能
                    "generationConfig": {"temperature": 0.6}
                }
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
                
                try:
                    res = call_api_with_retry(url, headers={'Content-Type': 'application/json'}, payload=payload)
                    if res and res.status_code == 200:
                        reply_text = res.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'エラー')
                        st.markdown(reply_text)
                        st.session_state.messages.append({"role": "assistant", "content": {"text": reply_text}})
                        st.session_state.api_history.extend([
                            {"role": "user", "parts": [{"text": prompt}]},
                            {"role": "model", "parts": [{"text": reply_text}]}
                        ])
                        st.rerun()
                    elif res:
                        if res.status_code == 503:
                            st.error("現在AIサーバーが大変混み合っております。何度か自動再試行しましたが接続できませんでした。少し時間をおいてから再度お試しください。")
                        else:
                            st.error(f"APIエラー: HTTP {res.status_code} - {res.text}")
                    else:
                        st.error("APIリクエストが失敗しました。")
                except Exception as e:
                    st.error(f"通信エラー: {e}")

# ------------------------------------------
# タブ2: 査定チェッカー (進化したプロンプトによるJSON出力とUI描画)
# ------------------------------------------
with tab2:
    st.markdown("### 🔍 商品の怪しさ・適正価格を鑑定します")
    st.write("商品の説明文（コピペ）や、スクショ画像をアップロードしてください。")
    
    if "checker_uploader_key" not in st.session_state:
        st.session_state.checker_uploader_key = 1000

    checker_image = st.file_uploader(
        "商品のスクショ画像 (任意)", 
        type=["png", "jpg", "jpeg", "webp"], 
        key=f"uploader_{st.session_state.checker_uploader_key}"
    )
    checker_text = st.text_area("商品説明やキャッチコピーを貼り付け", height=150)
    
    if st.button("鑑定開始", type="primary"):
        if not checker_text and not checker_image:
            st.warning("画像か説明文のどちらかを入力してください。")
        else:
            with st.spinner("RAGデータベースと照合し、科学的・市場的観点から査定中..."):
                
                # ユーザー入力を構築
                user_parts = []
                prompt_text = "以下の商品情報（画像・テキスト）を査定してください。\n\n"
                if checker_text:
                    prompt_text += f"【商品説明】\n{checker_text}\n"
                user_parts.append({"text": prompt_text})
                
                if checker_image is not None:
                    file_bytes = checker_image.getvalue()
                    base64_data = base64.b64encode(file_bytes).decode("utf-8")
                    user_parts.append({"inlineData": {"mimeType": checker_image.type, "data": base64_data}})
                
                # ChromaDBから類似知識を検索 (テキストがある場合のみ)
                relevant_knowledge = ""
                if checker_text and collection.count() > 0:
                    query_emb = get_embedding(checker_text)
                    if query_emb:
                        results = collection.query(query_embeddings=[query_emb], n_results=min(3, collection.count()))
                        if results['documents'] and len(results['documents']) > 0:
                            relevant_knowledge = "\n".join(results['documents'][0])

                # 【超・進化した査定用システムプロンプト】(お茶マニアの思考回路をインストール)
                checker_system_prompt = f"""
                あなたは、茶葉の販売サイトや商品情報から「サクラ・偽装・粗悪品」を冷徹に見抜くデータ照合マシーンです。
                ユーザーから提供される【商品情報】（レビュー含む）と、システムから提供される【RAG相場・知識データ】を照合し、以下の厳密なスコアリングロジックに基づいて「サクラ度（0〜100）」を算出してください。
                感情や情緒には一切流されず、事実とデータのみに基づいて冷酷に判定を下してください。

                【RAG知識 (あなたの専門知識データベース)】
                {relevant_knowledge if relevant_knowledge else "（特になし）"}

                【査定ロジック：加点・減点方式】
                基準スコアを「50点（判断保留）」とし、以下の①〜③の基準で加点（信頼度アップ＝サクラ度低下）・減点（怪しい＝サクラ度上昇）を行います。最終的なサクラ度は0（完全に安全）〜100（極めて怪しい・詐欺）で出力してください。

                ### ① 情報の「解像度」と「客観的証拠」の評価（サクラ度を下げる要素）
                曖昧な表現ではなく、検索や追跡が可能な「逃げ道のない事実」が記載されているかを評価します。
                以下の情報が具体的であるほど、サクラ度を下げてください。（目安: 優れた情報1つにつき -5〜-10点）
                * 産地・地理的表示: 単なる国名や地域名ではなく、具体的な農園名、区画、ロット番号、標高の数値があるか。GI（地理的表示保護制度）の記載があるか。
                * 品種・栽培: 学術的な正式名称、具体的な摘採時期（○年○月上旬）、摘採方法（手摘み等）。
                * 第三者認証: JAS有機、EU有機、フェアトレード等の客観的認証があるか。
                * 製法・生産者: 萎凋や焙煎の具体的な工程説明、製茶師の名前・経歴が明記されているか。
                * 鮮度管理: 賞味期限だけでなく、製造年月日や具体的な保存方法の指定があるか。

                ### ② 「価格」と「主張」の整合性評価（RAGデータとの照合）
                謳い文句と、RAGから提供される相場データ（産地、品種、摘採時期、製法などの多次元データ）を比較し、矛盾を突きます。
                * 安すぎる矛盾（目安: +20〜+30点）: 「最高級」「手摘み」と謳っているのに相場より著しく安い場合、香料添加、低品質茶葉のブレンド、偽装表示の可能性が高いと断定すること。
                * 高すぎる矛盾（目安: +15〜+25点）: ①の「具体的な事実（農園名や品種など）」がスッカスカであるにもかかわらず高価格な場合。「情弱向けのぼったくりビジネス」と判断すること。
                * 正当な高価格（目安: -10〜-20点）: ①の客観的証拠（単一農園、古樹、明確な製法）が網羅的に提示されており、それがRAGの「最高級相場」と一致する場合は「適正価格」とみなす。

                ### ③ 「ごまかし」と「不誠実さ」の検知（サクラ度を急上昇させる要素）
                販売者が「お茶の事実」以外で売ろうとするシグナルを検知し、強く減点（サクラ度上昇）してください。
                * 情緒的ワードの多用（目安: +5〜+15点）: 「至福の」「奇跡の」「癒しの」といった客観性のない形容詞が説明の中心を占めている。
                * 健康効果の過剰・違法な主張（目安: +25〜+30点）: 「ダイエットに！」「ガン予防」「血糖値が下がる」など、薬機法に抵触する表現や、科学的根拠（論文等）のない効能を謳っている。
                * 煽り文句（目安: +10〜+15点）: 「ランキング1位！」「今だけ半額！」「有名人も愛用！」などの過剰な煽り。
                * 不自然なレビュー（目安: +15〜+25点）: 具体性がなく高評価ばかり、短期間に集中している、機械翻訳のような不自然な日本語、他商品と同じ定型文の使い回しなどのサクラレビューの兆候。
                * 販売者の透明性欠如（目安: +10〜+20点）: 企業情報（所在地、代表者）が不明瞭、不自然な日本語。

                【出力フォーマット（JSON形式）】
                必ず以下のJSON形式のみで出力してください。Markdownのコードブロック(```json)は使用せず、純粋なJSON文字列のみを出力すること。出力するコメントは以下の例のように具体的かつ冷徹に記述してください。
                {{
                  "sakura_score": [最終的なサクラ度を0〜100の整数で出力。100が最も怪しい],
                  "risk_level": "[安全 / 注意 / 危険 のいずれかを出力]",
                  "analysis_details": {{
                    "resolution_check": "[例: 産地は『台湾・阿里山』とあるが、具体的な農園名、標高、ロット番号の記載がなく情報解像度が低い。品種も『烏龍茶』と曖昧で、特定の品種名が不明な点は大きな不足。第三者認証の記載も一切ない。]",
                    "price_consistency": "[例: 『最高級手摘み』を謳いながら100g 500円は、RAG相場データ（同等スペックで最低3000円）と著しく乖離している。香料添加や低品質茶葉のブレンドによる偽装の可能性が極めて高い。]",
                    "deception_signals": "[例: 『ガン予防に効く』という薬機法抵触の記述あり。また、『至福の香り』等の情緒的ワードが多用されており事実の提示が乏しい。レビューも短期間に『家族が喜んだ』という定型文が集中しており不自然。]"
                  }},
                  "conclusion": "[例: 客観的な事実が欠如しているにもかかわらず、不当に高い健康効果と安価な価格を提示しており、典型的な粗悪品・サクラのパターンに合致する。購入は強く非推奨。]"
                }}
                """

                payload = {
                    "systemInstruction": {"parts": [{"text": checker_system_prompt}]},
                    "contents": [{"role": "user", "parts": user_parts}],
                    # JSONモード(responseMimeType: application/json)とツール(google_search)は併用不可のため削除
                    "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
                }
                
                url = f"[https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=](https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=){API_KEY}"
                
                try:
                    res = call_api_with_retry(url, headers={'Content-Type': 'application/json'}, payload=payload)
                    if res and res.status_code == 200:
                        result_text = res.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')
                        
                        try:
                            # AIが返してきたJSONをPythonで解釈する
                            assessment = json.loads(result_text)
                            
                            # --- UIの描画 (Streamlitの魔法) ---
                            st.markdown("### 📊 鑑定結果")
                            
                            # 危険度スコアをメーターと色で表示
                            score = assessment.get("sakura_score", 0)
                            level = assessment.get("risk_level", "不明")
                            
                            st.write(f"**危険度（サクラ度）スコア:** {score}/100")
                            # Streamlitのプログレスバー機能
                            st.progress(score / 100.0) 
                            
                            if level == "危険":
                                st.error(f"🚨 【危険度：高】 この商品は誇大広告や不当価格の可能性が極めて高いです。")
                            elif level == "注意":
                                st.warning(f"⚠️ 【危険度：中】 いくつか疑わしい点があります。購入は慎重に。")
                            else:
                                st.success(f"✅ 【危険度：低】 特に怪しい点・非科学的な記述は見当たりません。")
                                
                            st.markdown("---")
                            
                            details = assessment.get("analysis_details", {})
                            
                            st.markdown("#### 🔎 ① 情報解像度チェック")
                            st.info(details.get("resolution_check", "情報なし"))
                            
                            st.markdown("#### 💰 ② 価格と主張の整合性")
                            st.warning(details.get("price_consistency", "情報なし"))
                            
                            st.markdown("#### 🎭 ③ ごまかしシグナル検知")
                            st.error(details.get("deception_signals", "情報なし"))
                            
                            st.markdown("#### 👨‍⚖️ 最終結論")
                            st.write(f"**{assessment.get('conclusion', '結論なし')}**")
                            
                            # 次の査定のためにファイルアップローダーのキーを更新
                            st.session_state.checker_uploader_key += 1
                            
                        except json.JSONDecodeError:
                            st.error("AIの判定結果のフォーマットエラーです。もう一度お試しください。")
                            st.write("生データ:", result_text)
                            
                    elif res:
                        if res.status_code == 503:
                            st.error("現在AIサーバーが大変混み合っております。何度か自動再試行しましたが接続できませんでした。少し時間をおいてから再度お試しください。")
                        else:
                            st.error(f"APIエラー: HTTP {res.status_code} - {res.text}")
                    else:
                        st.error("APIリクエストが失敗しました。")
                except Exception as e:
                    st.error(f"通信エラー: {e}")
