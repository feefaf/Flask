var jason =[
{
    "dara": 2,
    "jaba":{
            "ui":3,
            "uu":5
            },
    "ro":5
  }
]


const fileSystem = require("browserify-fs")

const data = JSON.stringify(client)

fileSystem.writeFile("./newClient.json", data, err=>{
 if(err){
   console.log("Error writing file" ,err)
 } else {
   console.log('JSON data is written to the file successfully')
 }
})


