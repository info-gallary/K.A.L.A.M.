let currentClient = null;

export const setClient = (ws) => {
  currentClient = ws;
};

export const getClient = () => {
  return currentClient;
};