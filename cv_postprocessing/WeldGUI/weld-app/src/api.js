import axios from 'axios';

 const API_BASE_URL = 'http://localhost:8080/api';    // For local development
//const API_BASE_URL = '/api';    // For deployment

// Patients
export const fetchPatients = () => {
    return axios.get(`${API_BASE_URL}/getPatients`);
};

export const fetchPatient = (patientId) => {
    return axios.get(`${API_BASE_URL}/getPatient?patientId=${patientId}`);
};

export const createPatient = (patient) => {
    return axios.post(`${API_BASE_URL}/createPatient`, patient);
};

export const updatePatient = (patient) => {
    return axios.put(`${API_BASE_URL}/updatePatient`, patient, {
        headers: { 'Content-Type': 'application/json' }
    });
};

export const deletePatient = (patientId) => {
    return axios.delete(`${API_BASE_URL}/deletePatient?patientId=${patientId}`);
};

export const createVisit = (visitData) => {
    return axios.post(`${API_BASE_URL}/visits`, visitData);
};

export const addSymptomsToVisit = (visitId, symptomIds) => {
    return axios.post(`${API_BASE_URL}/visits/${visitId}/symptoms`, { symptomIds });
};
export const fetchSymptoms = () => {
    return axios.get(`${API_BASE_URL}/getSymptoms`);
};
export const fetchInsuranceCompanies = () => {
    return axios.get(`${API_BASE_URL}/getInsuranceCompanies`);
};