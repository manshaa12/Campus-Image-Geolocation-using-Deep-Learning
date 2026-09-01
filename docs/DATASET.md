# Dataset notes

The project expects an image geolocation dataset with at least these columns:

```text
image, latitude, longitude
```

The public dataset used by default is:

```text
yyss114/CIS-5190-project-6
```

For local reference evaluation, a CSV file can use any of these common coordinate names:

```text
latitude / longitude
Latitude / Longitude
lat / lon
```

Image path columns can be named:

```text
image
image_path
filepath
path
file_name
filename
```

Raw images and full GPS data should not be committed to GitHub.
