import React from 'react';
import MapComponent from '../components/MapComponent';

const OverlayClouds = () => {
  return (
    <div className=" flex items-center justify-center w-full h-screen  bg-zinc-900">
      <div className="w-[600px] h-[600px]  rounded shadow ">
        <MapComponent />
      </div>
    </div>
  );
};

export default OverlayClouds;
