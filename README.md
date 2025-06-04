# 📚 Children's Book Style Converter

Transform classic books into delightful children's stories in the style of beloved authors like **Dr. Seuss**, **Beatrix Potter**, **Shel Silverstein**, and more.

This project uses LLMs to analyze and rewrite the chapters of uploaded PDF books into child-friendly narratives suitable for ages 5–10, removing scary or adult content and adapting the language to be engaging and appropriate.

---

## ✨ Features

- 🧠 **LLM-powered Rewriting**: Converts complex or adult content into child-appropriate language and style.
- ✍️ **Famous Author Styles**: Choose from popular children's authors for unique storytelling voices.
- 📄 **PDF Chapter Extraction**: Upload any PDF book and select specific chapters to transform.
- 💬 **Gradio Interface**: Easy-to-use web UI for interacting with the model.
- 📚 **Vector Store Integration**: Enables semantic search and context-aware rewriting.
- 🔍 **Chunked Processing**: Handles large chapters by breaking them into manageable parts.

---

## 🚀 Demo

> Coming soon via Hugging Face or Gradio Share URL.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Avani-Brahmbhatt/easybooks.git
   cd easybooks
   ```





2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file and add your Groq API key:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   MODEL_NAME=llama3-70b-8192  # or any other supported model
   ```

---

## 📦 Running the App

```bash
python app.py
```

This will launch the Gradio interface in your browser.

---

## 🧙 Author Styles

You can choose from the following author styles (customizable in `AUTHOR_STYLES` dictionary):

* Dr. Seuss
* Beatrix Potter
* Roald Dahl
* Mo Willems
* Shel Silverstein
* Arnold Lobel

---

Happy storytelling! 📖