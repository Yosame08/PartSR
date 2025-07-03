import FactoryMaker from './core/FactoryMaker.js';
import DashJSError from './streaming/vo/DashJSError.js';
import Errors from './core/errors/Errors.js';

function SR_API() {
    const localBackend = 'http://localhost:5555/';

    let idx = 0;
    let instance;

    function sr_api(api, data, filename, to, report){
        if (api === 'header') {
            report(data);
            const initFormData = new FormData();
            initFormData.append('metadata', JSON.stringify({ idx: 0 }, { type: 'application/json' }));
            initFormData.append('file', new Blob([data], { type: 'application/octet-stream' }), filename);
            fetch(localBackend + 'header', {method: 'POST', body: initFormData,}).catch(error => {console.log(error)});
            return;
        }
        if (api !== 'sr') {throw new Error('Unknown API');}

        const dataView0 = new DataView(data);
        const originLength = dataView0.getInt32(0, false);
        console.log(originLength);
        const originData = data.slice(4, 4 + originLength);
        report(originData);

        let offset = 4 + originLength; // 偏移量移到BIII部分
        const dataType = dataView0.getInt8(offset);
        offset += 1;

        let sendContent = (dataType === 0) ? data.slice(offset) : data

        const initFormData = new FormData();
        initFormData.append('metadata', JSON.stringify({ idx: ++idx }, { type: 'application/json' }));
        initFormData.append('file', new Blob([sendContent], { type: 'application/octet-stream' }), filename);

        fetch(localBackend + (dataType === 0 ? 'metric' : 'sr'), {
            method: 'POST',
            body: initFormData,
        }).catch(error => {
            console.log(error)
        });
    }

    instance = {
        sr_api,
    };

    return instance;
}

SR_API.__dashjs_factory_name = 'SR_API';
export default FactoryMaker.getSingletonFactory(SR_API);
