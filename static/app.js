const video = document.getElementById("video");

navigator.mediaDevices.getUserMedia({
    video: true
})
.then(stream => {
    video.srcObject = stream;
});

async function takePhoto(){

    const canvas = document.createElement("canvas");

    canvas.width = 224;
    canvas.height = 224;

    const ctx = canvas.getContext("2d");

    ctx.drawImage(video,0,0,224,224);

    canvas.toBlob(async(blob)=>{

        const formData = new FormData();

        formData.append(
            "file",
            blob,
            "image.jpg"
        );

        const response = await fetch(
            "http://127.0.0.1:8000/predict",
            {
                method:"POST",
                body:formData
            }
        );

        const data = await response.json();

        document.getElementById("result").innerText =
            data.label +
            " (" +
            (data.confidence*100).toFixed(2) +
            "%)";
    });
}