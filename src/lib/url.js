export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

export function toAbsoluteUrl(value) {
  if (!value || typeof value !== 'string') return null
  if (/^https?:\/\//i.test(value)) return value
  const normalized = value.startsWith('/') ? value : `/${value}`
  return `${API_BASE_URL}${normalized}`
}

