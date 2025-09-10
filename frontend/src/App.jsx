import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import MainLandingPage from './pages/MainLandingPage'
import ReportPage from './pages/ReportPage'
import OverlayClouds from './pages/OverlayClouds'
import SatelliteAnimationPage from './pages/SatelliteAnimationPage'
import { Toaster } from 'react-hot-toast';


const App = () => {
  return (
    <>
      <Toaster  position="top-center" reverseOrder={false} />
      <Router>
        <Routes>
          <Route path="/" element={<MainLandingPage />} />
          <Route path="/test" element={<ReportPage />} />
          <Route path="/overlay-clouds" element={<OverlayClouds />} />
          <Route path="/satellite-animation" element={<SatelliteAnimationPage />} />
        </Routes>
      </Router>
    </>
  )
}

export default App