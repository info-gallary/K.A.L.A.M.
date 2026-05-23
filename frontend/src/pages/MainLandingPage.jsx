import LandingPage from "../components/LandingPage";
import Beams from "../components/Beams";

const MainLandingPage = () => {
  return (
    <>
      {/* Indian Flag Background - 3 Vertical Sections */}
      <div className="fixed inset-0 w-full h-full z-0">
        {/* Saffron Section - Left */}
        <div className="absolute top-0 left-0 w-1/3 h-full bg-orange-600">
          <Beams
            beamWidth={3}
            beamHeight={30}
            beamNumber={20}
            lightColor="#FF9933"
            speed={3}
            noiseIntensity={1.75}
            scale={0.2}
            rotation={30}
          />
        </div>

        {/* White Section - Center */}
        <div className="absolute top-0 left-1/3 w-1/3 h-full bg-white">
          <Beams
            beamWidth={3}
            beamHeight={30}
            beamNumber={20}
            lightColor="#FFFFFF"
            speed={3}
            noiseIntensity={1.75}
            scale={0.2}
            rotation={30}
          />
        </div>

        {/* Green Section - Right */}
        <div className="absolute top-0 left-2/3 w-1/3 h-full bg-green-600">
          <Beams
            beamWidth={3}
            beamHeight={30}
            beamNumber={20}
            lightColor="#138808"
            speed={3}
            noiseIntensity={1.75}
            scale={0.2}
            rotation={30}
          />
        </div>
      </div>

      {/* LandingPage Content - Overlaid on top of all sections */}
      <div className="relative z-10">
        <LandingPage />
      </div>
    </>
  );
};

export default MainLandingPage;