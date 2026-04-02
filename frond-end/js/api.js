/**
 * API Wrapper for Skin AI Microservices
 * Encapsulates fetch logic and Auth Token management.
 */

const API_BASE = ""; // Relative since Traefik handles reverse proxy on port 80 or similar

const auth = {
  token: localStorage.getItem("api_token") || "",
  
  setToken(t) {
    this.token = t;
    if (t) {
      localStorage.setItem("api_token", t);
    } else {
      localStorage.removeItem("api_token");
    }
    // Dispatch custom event for UI to react
    window.dispatchEvent(new CustomEvent('authStateChanged', { detail: { isAuthenticated: !!t } }));
  },
  
  logout() {
    this.setToken("");
  },

  headers() {
    const headers = {
      "Accept": "application/json"
    };
    if (this.token) {
      headers["Authorization"] = `Bearer ${this.token}`;
    }
    return headers;
  },
  
  isAuthenticated() {
    return !!this.token;
  }
};

/**
 * Helper to handle fetch responses and errors consistently
 */
async function apiCall(endpoint, options = {}) {
  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers: {
        ...auth.headers(),
        ...(options.headers || {})
      }
    });

    if (!response.ok) {
      const errorText = await response.text();
      let errMsg = `HTTP Error ${response.status}`;
      try {
        const jsonExp = JSON.parse(errorText);
        if (jsonExp.detail) errMsg = typeof jsonExp.detail === 'string' ? jsonExp.detail : JSON.stringify(jsonExp.detail);
      } catch (e) {
        if (errorText) errMsg = errorText;
      }
      throw new Error(errMsg);
    }

    // Some endpoints might return empty body on Success (e.g. 204 No Content)
    const text = await response.text();
    return text ? JSON.parse(text) : null;

  } catch (error) {
    console.error(`[API Error] ${endpoint}:`, error);
    throw error; // Re-throw to be handled by UI
  }
}

// ------------------------------------------------------------------
// Auth Endpoints
// ------------------------------------------------------------------

async function login(username, password) {
  const formData = new FormData();
  formData.append("username", username);
  formData.append("password", password);

  const data = await apiCall("/auth/login", {
    method: "POST",
    body: formData // Login explicitly expects form-data according to README
  });
  
  if (data.access_token) {
    auth.setToken(data.access_token);
  }
  return data;
}

async function register(email, password) {
  return await apiCall("/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password })
  });
}

async function verifyAuth() {
  if (!auth.token) return false;
  try {
    await apiCall("/auth/verify");
    return true;
  } catch (err) {
    auth.logout();
    return false;
  }
}

// ------------------------------------------------------------------
// Patients Endpoints
// ------------------------------------------------------------------

async function getPatients() {
  return await apiCall("/api/v1/patients");
}

async function createPatient(name, notes) {
  return await apiCall("/api/v1/patients", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, notes })
  });
}

// ------------------------------------------------------------------
// Analyses Endpoints
// ------------------------------------------------------------------

async function getAnalyses() {
  return await apiCall("/api/v1/analyses");
}

async function createAnalysis(patientId, imageFile) {
  const formData = new FormData();
  formData.append("patient_id", patientId);
  formData.append("image", imageFile);

  return await apiCall("/api/v1/analyses", {
    method: "POST",
    body: formData
  });
}

// ------------------------------------------------------------------
// Direct ML Predictions
// ------------------------------------------------------------------

/**
 * Direct ML Prediction (used on landing page w/o auth usually)
 * Note README: "POST /ml/api/predict/ (multipart image)"
 */
async function predictImage(imageFile, onProgress) {
  const formData = new FormData();
  formData.append("image", imageFile);

  // We use XMLHttpRequest here to get upload progress for the UI
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    
    // Construct exactly as the original frontend did
    let url = "/ml/api/predict/";
    if (location.protocol === "file:") {
      url = "http://127.0.0.1:8000/api/predict/";
    }

    xhr.open("POST", url, true);
    
    // Add auth token if we happen to have it, just in case
    if (auth.token) {
        xhr.setRequestHeader("Authorization", `Bearer ${auth.token}`);
    }

    if (onProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          onProgress((e.loaded / e.total) * 100);
        }
      };
    }

    xhr.onload = function() {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const res = JSON.parse(xhr.responseText);
          resolve(res);
        } catch (e) {
          reject(new Error("Invalid JSON response"));
        }
      } else {
        reject(new Error(`Server error: ${xhr.status} ${xhr.responseText}`));
      }
    };

    xhr.onerror = function() {
      reject(new Error("Network error during prediction upload."));
    };

    xhr.send(formData);
  });
}

// Register globals
window.api = {
  auth,
  login,
  register,
  verifyAuth,
  getPatients,
  createPatient,
  getAnalyses,
  createAnalysis,
  predictImage
};
