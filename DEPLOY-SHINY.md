### Shinyapps:
*https://docs.posit.co/shinyapps.io/guide/getting_started/*

### Shinylive:
```bash
shinylive export ./peregrin_app docs
```



<!-- 

html:


<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Peregrin</title>
    <script
      src="./shinylive/load-shinylive-sw.js"
      type="module"
    ></script>
    <script type="module">
      import { runExportedApp } from "./shinylive/shinylive.js";
      runExportedApp({
        id: "root",
        appEngine: "python",
        relPath: "",
      });
    </script>
    <link rel="stylesheet" href="./shinylive/style-resets.css" />
    <link rel="stylesheet" href="./shinylive/shinylive.css" />
    <style>
      /* Force square corners and remove any default layout spacing */
      html, body, #root, #root iframe {
        margin: 0 !important;
        padding: 0 !important;
        border-radius: 0px !important;
      }
      #root > div {
        border-radius: 0px !important;
      }
    </style>
  </head>
  <body style="zoom: 80%; margin: 0; border-radius: 0px;">
    
    <div style="height: 125vh; width: 125vw; border-radius: 0px !important;" id="root"></div>
    
  </body>
</html>

 -->
