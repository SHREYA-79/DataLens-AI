with st.spinner("AI is analyzing your data..."):
    try:

        client = Groq(
            api_key=st.secrets["GROQ_API_KEY"]
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=2000
        )

        answer = response.choices[0].message.content

        st.success("Analysis complete")

        st.markdown(
            f'<div class="insight-card"><p>{answer}</p></div>',
            unsafe_allow_html=True
        )

        buf = io.StringIO()
        buf.write(
            f"DataLens AI\nQuestion: {question}\n\n{answer}"
        )

        st.download_button(
            "⬇️ Download analysis",
            buf.getvalue(),
            file_name="datalens_ai_analysis.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"AI error: {e}")
