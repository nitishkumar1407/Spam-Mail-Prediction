import tkinter as tk
from tkinter import ttk, font
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def load_model():
    # Load dataset
    raw_mail_data = pd.read_csv(r"/Users/adityagarg/Library/Containers/net.whatsapp.WhatsApp/Data/tmp/documents/0E4E1D42-5707-434C-944D-E445D70FB351/mail_data.csv")
    # Handle missing values
    mail_data = raw_mail_data.where(pd.notnull(raw_mail_data))
    
    # Encode labels
    mail_data['Category'] = mail_data['Category'].map({'spam': 0, 'ham': 1})
    
    # Split data
    x = mail_data['Message']
    y = mail_data['Category']
    
    x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.9, random_state=3)
    
    # Feature extraction
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    x_train_features = feature_extraction.fit_transform(x_train)
    x_test_features = feature_extraction.transform(x_test)
    
    # Train model
    model = LogisticRegression()
    model.fit(x_train_features, y_train)
    
    return model, feature_extraction

# Load the model and vectorizer when the app starts
model, feature_extraction = load_model()

def predict_spam():
    message = text_box.get("1.0", "end-1c")
    if message.strip():  
        
        input_data_features = feature_extraction.transform([message])
        
       
        prediction = model.predict(input_data_features)
        
        result = "Spam" if prediction[0] == 0 else "Ham"
        result_label.config(text=f"Prediction: {result}", fg="red" if result == "Spam" else "green")
    else:
        result_label.config(text="Please enter a message", fg="black")

# Create the GUI
root = tk.Tk()
root.title("Spam Detector ")

# Custom fonts
title_font = font.Font(family='Helvetica', size=16, weight='bold')
subtitle_font = font.Font(family='Helvetica', size=12)
input_font = font.Font(family='Arial', size=10)
button_font = font.Font(family='Helvetica', size=10, weight='bold')
footer_font = font.Font(family='Helvetica', size=8)

# Main frame
main_frame = ttk.Frame(root, padding="20")
main_frame.pack(fill=tk.BOTH, expand=True)

# Title
title_label = ttk.Label(main_frame, text="Spam Detector ", font=title_font)
title_label.pack(pady=(0, 10))

# Subtitle
subtitle_label = ttk.Label(main_frame, text="Spam Detector For Email Messages", font=subtitle_font)
subtitle_label.pack(pady=(0, 15))

# Instruction
instruction_label = ttk.Label(main_frame, 
                             text="Enter Your Message/Mail Here which you want to predict\nSpam or Ham",
                             font=input_font)
instruction_label.pack(pady=(0, 10))

# Text box with placeholder
text_box = tk.Text(main_frame, width=60, height=10, wrap=tk.WORD, font=input_font,
                  borderwidth=1, relief="solid", padx=5, pady=5)
text_box.pack(pady=(0, 15))
text_box.insert("1.0", "Enter Your Message...")
text_box.bind("<FocusIn>", lambda args: text_box.delete("1.0", "end") if text_box.get("1.0", "end-1c") == "Enter Your Message..." else None)

# Predict button
predict_button = ttk.Button(main_frame, text="Predict", command=predict_spam, style='TButton')
predict_button.pack(pady=(0, 15))

# Result label
result_label = tk.Label(main_frame, text="", font=button_font)
result_label.pack()

# Footer
footer_frame = ttk.Frame(main_frame)
footer_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(20, 0))

footer_label = ttk.Label(footer_frame, text="Made by Aiml syndicate", font=footer_font)
footer_label.pack(side=tk.LEFT)

version_label = ttk.Label(footer_frame, text="1.002x55", font=footer_font)
version_label.pack(side=tk.RIGHT)

# Style configuration
style = ttk.Style()
style.configure('TButton', font=button_font)

# Center the window
root.eval('tk::PlaceWindow . center')

root.mainloop()