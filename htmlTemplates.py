css = '''
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
    align-items: center;  /* Ensures vertical alignment */
}

.chat-message.user {
    background-color: #333c76;
}

.chat-message.bot {
    background-color: #ac848a;
}

.chat-message .avatar {
  width: 50px; /* Set a fixed width */
  height: 50px; /* Set a fixed height */
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 10px; /* Add spacing between avatar and text */
}

.chat-message .avatar img {
  width: 50px;  /* Ensure consistent width */
  height: 50px; /* Ensure consistent height */
  border-radius: 50%;
  object-fit: cover; /* Ensures the image fits inside the avatar box */
}

.chat-message .message {
  flex-grow: 1; /* Allow the message box to take remaining space */
  padding: 0 1.5rem;
  color: #fff;
  word-wrap: break-word; /* Prevent text from overflowing */
}
</style>
'''

user_template = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
</div>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="message">{{MSG}}</div>
</div>
'''

