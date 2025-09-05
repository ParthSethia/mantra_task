import gradio as gr

# Dummy function to "analyze" video and return a placeholder result
def analyze_video(video, question, chat_history):
    if video is None:
        return chat_history + [("User: " + question, "Please upload a video first.")]
    
    # Placeholder logic (replace with actual video analysis)
    response = f"I analyzed the video and here’s an answer to: '{question}'"
    
    chat_history.append(("User: " + question, response))
    return chat_history


with gr.Blocks(title="Video Analysis Chatbot") as demo:
    gr.Markdown("## 🎥 Video Analysis Chatbot")
    gr.Markdown("Upload a video and ask questions about it!")

    with gr.Row():
        video_input = gr.Video(label="Upload Video")
    
    chatbot = gr.Chatbot(label="Conversation")
    question_input = gr.Textbox(placeholder="Ask a question about the video...", label="Your Question")
    submit_btn = gr.Button("Ask")

    state = gr.State([])  # Store chat history

    submit_btn.click(
        fn=analyze_video,
        inputs=[video_input, question_input, state],
        outputs=chatbot
    )

    question_input.submit(
        fn=analyze_video,
        inputs=[video_input, question_input, state],
        outputs=chatbot
    )

if __name__ == "__main__":
    demo.launch()
