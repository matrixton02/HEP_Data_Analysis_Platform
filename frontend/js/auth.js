// API base URL
const API_URL = 'http://localhost:8000/api';

// Show alert message
function showAlert(containerId, message, type = 'error') {
    const container = document.getElementById(containerId);
    const alertClass = type === 'error' ? 'alert-error' : 'alert-success';
    
    container.innerHTML = `
        <div class="alert ${alertClass}">
            ${message}
        </div>
    `;
    
    setTimeout(() => {
        container.innerHTML = '';
    }, 5000);
}

// Login form handler
document.getElementById('loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    try {
        const response = await fetch(`${API_URL}/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password }),
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Save token and user info
            localStorage.setItem('token', data.access_token);
            localStorage.setItem('userId', data.user_id);
            localStorage.setItem('username', data.username);
            
            // Redirect to home
            window.location.href = '/home';
        } else {
            showAlert('alertContainer', data.detail || 'Login failed', 'error');
        }
    } catch (error) {
        showAlert('alertContainer', 'Network error. Please try again.', 'error');
        console.error('Login error:', error);
    }
});

// Register form handler
document.getElementById('registerForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const username = document.getElementById('regUsername').value;
    const email = document.getElementById('regEmail').value;
    const password = document.getElementById('regPassword').value;
    
    try {
        const response = await fetch(`${API_URL}/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, email, password }),
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Save token and user info
            localStorage.setItem('token', data.access_token);
            localStorage.setItem('userId', data.user_id);
            localStorage.setItem('username', data.username);
            
            // Redirect to home
            window.location.href = '/home';
        } else {
            showAlert('registerAlertContainer', data.detail || 'Registration failed', 'error');
        }
    } catch (error) {
        showAlert('registerAlertContainer', 'Network error. Please try again.', 'error');
        console.error('Registration error:', error);
    }
});

// Toggle between login and register
document.getElementById('showRegister').addEventListener('click', () => {
    document.getElementById('loginForm').closest('.login-card').style.display = 'none';
    document.getElementById('registerModal').style.display = 'flex';
});

document.getElementById('showLogin').addEventListener('click', () => {
    document.getElementById('registerModal').style.display = 'none';
    document.getElementById('loginForm').closest('.login-card').style.display = 'block';
});

// Check if already logged in
if (localStorage.getItem('token')) {
    window.location.href = '/home';
}
