import axios from 'axios';

const api = axios.create({
    baseURL: "http://localhost:3005/api/",
    // withCredentials: true,
});

export const googleAuth = (code) => api.get(`/google?code=${code}`);