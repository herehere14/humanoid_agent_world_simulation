import React, { lazy, Suspense } from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

const WorldViewer = lazy(() => import('./world-viewer/WorldViewer'))

function Root() {
  // Simple hash-based routing: #/world → WorldViewer, else → App
  const [route, setRoute] = React.useState(window.location.hash)

  React.useEffect(() => {
    const handler = () => setRoute(window.location.hash)
    window.addEventListener('hashchange', handler)
    return () => window.removeEventListener('hashchange', handler)
  }, [])

  if (route === '#/world') {
    return (
      <Suspense fallback={
        <div style={{
          position: 'fixed', inset: 0, display: 'flex',
          alignItems: 'center', justifyContent: 'center',
          background: '#f8f9fb', fontFamily: 'Inter, system-ui, sans-serif',
        }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{
              width: 32, height: 32, border: '3px solid #e2e8f0',
              borderTopColor: '#0066ff', borderRadius: '50%',
              margin: '0 auto', animation: 'spin 0.8s linear infinite',
            }} />
            <p style={{ marginTop: 12, color: '#64748b', fontSize: 13 }}>
              Loading World Viewer...
            </p>
          </div>
        </div>
      }>
        <WorldViewer />
      </Suspense>
    )
  }

  return <App />
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
)
