import streamlit as st

# --- Simple Hardcoded User Data (Replace with a real database like Firebase for production) ---
USERS = {
    "user@example.com": "securepassword123",
    "admin": "adminpass"
}

def initialize_session_state():
    """Ensures essential session state keys are initialized."""
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'user' not in st.session_state:
        st.session_state['user'] = None

def login_page():
    """Renders the login/signup UI and handles authentication."""
    # Initialize state before using it
    initialize_session_state()

    st.sidebar.title("Cloud Cost Engine")

    st.title("Welcome! Sign In to Continue")
    st.caption("This is a mock authentication system using Streamlit Session State.")

    # Determine if user is logging in or signing up
    auth_mode = st.radio("Choose Mode", ["Login", "Sign Up"], index=0, horizontal=True)

    with st.form("auth_form"):
        email = st.text_input("Email/Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button(auth_mode)

        if submitted:
            if auth_mode == "Login":
                if email in USERS and USERS[email] == password:
                    st.session_state['logged_in'] = True
                    st.session_state['user'] = email
                    st.success(f"Welcome back, {email}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

            elif auth_mode == "Sign Up":
                if email in USERS:
                    st.error("User already exists. Please login or use a different email.")
                else:
                    # Mock registration
                    USERS[email] = password
                    st.session_state['logged_in'] = True
                    st.session_state['user'] = email
                    st.success(f"Account created and logged in! Welcome, {email}.")
                    st.rerun()

def logout():
    """Logs the user out and resets the session state."""
    if st.session_state.get('logged_in'):
        st.session_state['logged_in'] = False
        st.session_state.pop('df', None) # Clear uploaded data
        st.success("Logged out successfully.")
        st.rerun()

def show_logout_button():
    """Displays the logout button in the sidebar."""
    # Check if the session state is initialized before checking 'logged_in'
    initialize_session_state()
    if st.session_state.get('logged_in'):
        st.sidebar.button("Logout", on_click=logout)
