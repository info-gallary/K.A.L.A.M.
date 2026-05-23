import { useState } from 'react';
import { Sun, Moon, ArrowUpRight, Cloud, BarChart3, Play,Map   } from 'lucide-react';
import RotatingText from './RotatingText';
import { Link } from 'react-router-dom';

const LandingPage = () => {
  const [darkMode, setDarkMode] = useState(true);

  const toggleTheme = () => {
    setDarkMode(!darkMode);
  };

  // Claude Brand Theme Colors
  const themeClasses = {
    // Claude's official colors: Crail (#C15F3C), Cloudy (#B1ADA1), Pampas (#F4F3EE), White (#FFFFFF)
    // Also using terra cotta from logo: #da7756
    bg: darkMode ? 'bg-stone-900/90' : 'bg-neutral-50/90',
    bgOverlay: darkMode 
      ? 'bg-gradient-to-br from-stone-900/50 via-transparent to-stone-800/30'
      : 'bg-gradient-to-br from-neutral-50/80 via-white/40 to-stone-50/60',
    nav: darkMode ? 'bg-stone-900/80 border-stone-700/30' : 'bg-white/80 border-stone-200',
    text: darkMode ? 'text-stone-100' : 'text-stone-900',
    textSecondary: darkMode ? 'text-stone-300' : 'text-stone-700',
    textMuted: darkMode ? 'text-stone-400' : 'text-stone-600',
    logo: darkMode ? 'bg-stone-600 text-stone-100' : 'bg-stone-600 text-stone-100',
    navButton: darkMode 
      ? 'text-stone-300 hover:bg-stone-800/70 hover:text-stone-100'
      : 'text-stone-700 hover:bg-stone-50 hover:text-stone-800',
    themeButton: darkMode
      ? 'text-stone-400 hover:bg-stone-800/70 hover:text-stone-100'
      : 'text-stone-600 hover:bg-stone-50 hover:text-stone-700',
    primaryButton: darkMode
      ? 'text-stone-900 hover:bg-orange-100'
      : 'text-white hover:opacity-90',
    secondaryButton: darkMode
      ? 'border-stone-600 text-stone-300 hover:bg-stone-800/70 hover:border-stone-500'
      : 'border-stone-300 text-stone-700 hover:bg-stone-50 hover:border-stone-400',
    // Claude brand colors
    accent: '#C15F3C', // Crail - Claude's primary brand color
    accentSecondary: '#da7756' // Terra cotta from logo
  };

  return (
    <div className={`min-h-screen transition-all duration-200 poppins-regular mozilla-headline-custom ${themeClasses.text}`}>
      {/* Subtle Background Overlay - Updated for Claude theme */}
      <div className={`fixed inset-0 pointer-events-none transition-opacity duration-200 ${themeClasses.bgOverlay}`} 
           style={{ zIndex: 1 }} />

      {/* Content */}
      <div className="relative" style={{ zIndex: 2 }}>
        {/* Claude-themed Navigation */}
        <nav className={`fixed top-0 w-full z-50 backdrop-blur-md border-b transition-all duration-200 ${themeClasses.nav}`}>
          <div className="max-w-6xl mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              {/* Logo with Claude styling */}
              <div className="flex items-center space-x-3">
                <div className={`size-12 rounded-lg flex items-center justify-center text-sm font-semibold ${themeClasses.logo}`}>
                  <img className="size-12 " src="sir_kalam.png" alt="APJ ABDUL KALAM" />
                </div>
                <span className="text-3xl font-semibold">DOMinators</span>
              </div>

              {/* Navigation Buttons */}
              <div className="flex items-center space-x-3">
                <Link
                  to='/overlay-clouds'
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 hover:scale-[1.02] ${themeClasses.navButton}`}
                >
                  <Map className="size-6" />
                  <span className='text-lg'>Visualize On Map</span>
                </Link>

                <Link
                  to="/test"
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 hover:scale-[1.02] ${themeClasses.navButton}`}
                >
                  <BarChart3 className="size-6" />
                  <span className='text-lg'>Test Model</span>
                </Link>

                <Link
                  to="/satellite-animation"
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 hover:scale-[1.02] ${themeClasses.navButton}`}
                >
                  {/* <Play className="size-6" />
                   */}
                   <Cloud className="size-6" />
                  <span className='text-lg'>Chase The Cloud</span>
                </Link>

                <button
                  onClick={toggleTheme}
                  className={`p-2 rounded-lg transition-all duration-200 hover:scale-[1.02] ${themeClasses.themeButton}`}
                  style={{ 
                    backgroundColor: darkMode ? 'transparent' : themeClasses.accent,
                    color: darkMode ? undefined : 'white'
                  }}
                >
                  {darkMode ? (
                    <Sun className="w-5 h-5" />
                  ) : (
                    <Moon className="w-5 h-5" />
                  )}
                </button>
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="pt-24 px-6">
          <div className="max-w-6xl mx-auto">
            <div className="min-h-[80vh] flex items-center justify-center">
              {/* Centered Content */}
              <div className="text-center space-y-12 max-w-4xl">

                {/* Main Heading */}
                <div className="space-y-6">
                  <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold leading-tight tracking-tight flex items-center justify-center flex-wrap gap-x-2">
                    <span>PROJECT</span>
                    <div className="inline-block w-32 md:w-40 lg:w-48">
                      <RotatingText
                        texts={['KALAM', 'कलाम', 'કલામ','கலாம்']}
                        mainClassName="px-2 sm:px-2 md:px-3 text-white overflow-hidden py-0.5 sm:py-1 md:py-2 justify-center rounded-lg text-3xl md:text-4xl lg:text-5xl inline-flex w-full"
                        staggerFrom={"last"}
                        initial={{ y: "100%" }}
                        animate={{ y: 0 }}
                        exit={{ y: "-120%" }}
                        staggerDuration={0.025}
                        splitLevelClassName="overflow-hidden pb-0.5 sm:pb-1 md:pb-1"
                        transition={{ type: "spring", damping: 30, stiffness: 400 }}
                        rotationInterval={2000}
                        style={{ backgroundColor: themeClasses.accent }}
                      />
                    </div>
                  </h1>

                  <p className={`text-lg md:text-xl leading-relaxed mx-auto max-w-2xl ${themeClasses.textSecondary}`}>
                   "Dream is not that which you see while sleeping, it is something that does not let you sleep."
                   - Dr. APJ Abdul Kalam
                  </p>
                </div>

                {/* CTA Section */}
                <div className="space-y-6">
                  <p className={`text-base ${themeClasses.textMuted}`}>
                    Experience the future of atmospheric prediction
                  </p>

                  <div className="flex flex-col sm:flex-row gap-3 justify-center">
                    {/* Primary Button - Claude Brand Color */}
                    <Link
                      to="/satellite-animation"
                      className={`text-white group flex items-center justify-center space-x-2 px-6 py-3 rounded-lg font-medium text-sm transition-all duration-200 hover:scale-[1.02] shadow-sm ${themeClasses.primaryButton}`}
                      style={{ 
                        backgroundColor: themeClasses.accent,
                        boxShadow: darkMode ? '0 1px 3px rgba(193, 95, 60, 0.3)' : '0 1px 3px rgba(0, 0, 0, 0.2)'
                      }}
                    >
                      <Play className="w-4 h-4 " />
                      <span>Get Started</span>
                      <ArrowUpRight className="w-4 h-4 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform duration-200" />
                    </Link>

                    {/* Secondary Button */}
                    <Link
                      to="/test"
                      className={`group flex items-center justify-center space-x-2 px-6 py-3 rounded-lg font-medium text-sm transition-all duration-200 hover:scale-[1.02] border ${themeClasses.secondaryButton}`}
                    >
                      <BarChart3 className="w-4 h-4" />
                      <span>View Reports</span>
                    </Link>
                  </div>
                </div>

              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default LandingPage;