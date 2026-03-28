import React, { lazy, Suspense } from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

const WorldViewer = lazy(() => import('./world-viewer/WorldViewer'))
const SimulationViewer = lazy(() => import('./simulate/SimulationViewer'))
const CascadeLanding = lazy(() => import('./simulate/CascadeLanding'))

function Root() {
  const [route, setRoute] = React.useState(window.location.hash)

  React.useEffect(() => {
    const handler = () => setRoute(window.location.hash)
    window.addEventListener('hashchange', handler)
    return () => window.removeEventListener('hashchange', handler)
  }, [])

  if (route === '#/simulate' || route.startsWith('#/simulate?')) {
    return (
      <Suspense fallback={
        <div style={{
          position: 'fixed', inset: 0, display: 'flex',
          alignItems: 'center', justifyContent: 'center',
          background: '#fafbfc', fontFamily: 'Inter, system-ui, sans-serif',
        }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{
              width: 32, height: 32, border: '3px solid #e2e8f0',
              borderTopColor: '#0066ff', borderRadius: '50%',
              margin: '0 auto', animation: 'spin 0.8s linear infinite',
            }} />
            <p style={{ marginTop: 12, color: '#64748b', fontSize: 13 }}>
              Loading Simulation Viewer...
            </p>
          </div>
        </div>
      }>
        <SimulationViewer />
      </Suspense>
    )
  }

  if (route === '#/world' || route.startsWith('#/world?')) {
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

  // Default: Cascade landing page (use #/legacy for old App)
  if (route === '#/legacy') {
    return <App />
  }

  return (
    <Suspense fallback={
      <div style={{ position: 'fixed', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#fafbfc' }}>
        <div style={{ width: 32, height: 32, border: '3px solid #e2e8f0', borderTopColor: '#3b82f6', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
      </div>
    }>
      <CascadeLanding />
    </Suspense>
  )
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
)
