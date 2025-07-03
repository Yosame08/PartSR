# PartSR

## Architecture

Before running the code, you can set your own configurations in `config.yaml`.

### Server

Run `server.py` to start the edge server.

It will automatically forward the request to the source server. If the request is for a video clip in dash format, it will be automatically processed and returned to the client, otherwise the original response is returned.

### Client

Run `client.py` to start the client.

It waits the modified `dash.js` to send response to the client. Then it will do SR tasks or compute metrics.

### Dash.js

The frontend of client. Copy `SR_API.js` to `dash.js` and modify `function load(request)` like this:

```javascript
success: function (data) {
    // if (false) {
    if (request.mediaType === 'video') {
        switch (request.type) {
            case 'InitializationSegment':
                sr_api.sr_api('header', data, request.url.split('/').pop(), request.representation.id, report);
                break;
            case 'MediaSegment':
                sr_api.sr_api('sr', data, request.url.split('/').pop(), request.representation.id, report);
                break;
        }
    } else {
        report(data);
    }
},
```

Set the source url in HTML files which uses `dash.js` to your source server.