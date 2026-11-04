import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
});

const DEFAULT_ADMIN_KEY = 'fruitvision-admin-dev-key-change-in-prod';
const ADMIN_API_KEY = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_ADMIN_KEY) || DEFAULT_ADMIN_KEY;

const getAdminHeaders = () => ({
  'X-API-Key': ADMIN_API_KEY,
  'X-Admin-Key': ADMIN_API_KEY,
});

export const getHealth = async () => {
  const res = await api.get('/health');
  return res.data;
};

export const reloadModel = async () => {
  const res = await api.post('/reload-model', {}, {
    headers: getAdminHeaders(),
  });
  return res.data;
};

export const getAnalyticsSummary = async () => {
  const res = await api.get('/analytics/summary');
  return res.data;
};

export const getModelMetrics = async () => {
  const res = await api.get('/analytics/model-metrics');
  return res.data;
};

export const getHistory = async (params = {}) => {
  const res = await api.get('/history', { params });
  return res.data;
};

export const deleteHistoryRecord = async (id) => {
  const res = await api.delete(`/history/${id}`, {
    headers: getAdminHeaders(),
  });
  return res.data;
};

export const purgeAllHistory = async () => {
  const res = await api.delete('/history', {
    headers: getAdminHeaders(),
  });
  return res.data;
};

export const inspectSingleImage = async (file, notes = '') => {
  const formData = new FormData();
  formData.append('file', file);
  if (notes) formData.append('notes', notes);

  const res = await api.post('/inspect/image', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export const inspectMultiFruit = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const res = await api.post('/inspect/multi', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export const inspectLiveFrame = async (frameBase64, sourceId = 'webcam') => {
  const res = await api.post('/inspect/frame', {
    frame_data: frameBase64,
    source_id: sourceId,
  });
  return res.data;
};

export const getSimulationSample = async (fruitChoice = 'random', mode = 'all') => {
  const params = { fruit_choice: fruitChoice };
  if (mode && mode !== 'all') {
    params.mode = mode;
  }
  const res = await api.get('/simulation/sample', { params });
  return res.data;
};

export const getSimulationStats = async () => {
  const res = await api.get('/simulation/stats');
  return res.data;
};

export const resetSimulation = async () => {
  const res = await api.post('/simulation/reset');
  return res.data;
};
