export const testModel = (req,res)=> {

  console.log("Model test function called");
  res.status(200).json({
    message: 'Model test successful',
    timestamp: new Date().toISOString(),
    data: {
      modelName: 'TestModel',
      status: 'active',
      version: '1.0.0'
    }
  })
}