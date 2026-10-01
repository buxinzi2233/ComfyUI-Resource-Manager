import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

async function request(path, method = "GET", body) {
    const options = { method };
    if (body !== undefined) {
        options.headers = { "Content-Type": "application/json" };
        options.body = JSON.stringify(body);
    }
    const response = await api.fetchApi(`/resource-manager/${path}`, options);
    if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || `请求失败（HTTP ${response.status}）`);
    }
    return response.json();
}

function element(tag, className, text) {
    const node = document.createElement(tag);
    node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

function button(id, text, icon) {
    const node = element("button", "rm-button");
    node.type = "button";
    node.id = id;
    if (icon) {
        const glyph = element("i", `pi ${icon}`);
        glyph.setAttribute("aria-hidden", "true");
        node.append(glyph);
    }
    node.append(element("span", "", text));
    return node;
}

function displayBytes(bytes) {
    if (bytes === null) return "—";
    const gib = bytes / 1024 ** 3;
    return gib >= 1 ? `${gib.toFixed(2)} GiB` : `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
}

function section(title) {
    const node = element("section", "rm-section");
    node.append(element("h3", "rm-section-title", title));
    return node;
}

function timerControl(id, title, description, defaultSeconds) {
    const row = element("div", "rm-timer");
    const label = element("label", "rm-toggle-label");
    const toggle = element("input", "rm-toggle");
    toggle.type = "checkbox";
    toggle.id = `${id}-enabled`;
    toggle.setAttribute("role", "switch");
    label.append(toggle, element("span", "", title));
    const field = element("div", "rm-seconds-field");
    const input = element("input", "rm-input");
    input.type = "number";
    input.id = id;
    input.min = "1";
    input.max = "86400";
    input.step = "1";
    input.required = true;
    input.value = String(defaultSeconds);
    input.setAttribute("aria-label", `${title}空闲秒数`);
    field.append(input, element("span", "rm-muted", "秒"));
    const hint = element("p", "rm-hint", description);
    hint.id = `${id}-hint`;
    input.setAttribute("aria-describedby", hint.id);
    row.append(label, field, hint);
    return { row, toggle, input, defaultSeconds };
}

function mount(container) {
    container.classList.add("rm-host");
    const root = element("div", "rm-panel");
    container.replaceChildren(root);
    let stopped = false;
    let poll;
    let latest = null;
    let dirty = false;
    let working = false;
    let refreshId = 0;
    const records = new Map();

    const header = element("header", "rm-header");
    header.append(element("h2", "rm-title", "资源管理"));
    const state = element("span", "rm-badge", "连接中");
    header.append(state);
    const connectionError = element("p", "rm-message rm-error");
    connectionError.hidden = true;
    connectionError.setAttribute("role", "alert");
    const actionMessage = element("p", "rm-message");
    actionMessage.hidden = true;
    actionMessage.setAttribute("role", "status");

    const memory = section("资源占用");
    const metrics = element("dl", "rm-metrics");
    const values = ["GPU 张量", "GPU 缓存池", "进程内存"].map((title) => {
        const row = element("div", "rm-metric");
        const value = element("dd", "rm-value", "—");
        row.append(element("dt", "rm-muted", title), value);
        metrics.append(row);
        return value;
    });
    const queue = element("p", "rm-hint", "正在读取运行状态…");
    memory.append(metrics, queue);

    const manual = section("手动释放");
    const unload = button("rm-unload-gpu", "卸载 GPU 模型", "pi-eject");
    const release = button("rm-release-models", "释放模型与缓存", "pi-trash");
    manual.append(unload, element("p", "rm-hint", "保留内存中的模型，下次加载更快。"),
        release, element("p", "rm-hint", "同时清除执行缓存，下次运行需重新加载。"));

    const automatic = section("自动释放");
    const autoStatus = element("p", "rm-auto-status", "正在读取设置…");
    autoStatus.id = "rm-auto-status";
    const gpu = timerControl("rm-gpu-seconds", "卸载 GPU 模型", "队列空闲后卸载显存中的模型。", 60);
    const full = timerControl("rm-full-seconds", "释放模型与缓存", "队列空闲后释放模型及执行缓存。", 300);
    const form = element("form", "rm-settings");
    const switchRow = element("div", "rm-timer");
    const switchLabel = element("label", "rm-toggle-label");
    const switchToggle = element("input", "rm-toggle");
    switchToggle.id = "rm-model-switch-enabled";
    switchToggle.type = "checkbox";
    switchToggle.setAttribute("role", "switch");
    switchLabel.append(switchToggle, element("span", "", "切换模型时释放旧模型"));
    const switchHint = element("p", "rm-hint", "下次运行前释放旧模型，保留当前工作流仍需的模型。支持原生 Checkpoint、UNet、CLIP、VAE 加载器。");
    switchHint.id = "rm-model-switch-hint";
    switchToggle.setAttribute("aria-describedby", switchHint.id);
    switchRow.append(switchLabel, switchHint);
    const save = button("rm-save-settings", "保存设置", "pi-check");
    save.type = "submit";
    const settingsState = element("span", "rm-muted", "");
    settingsState.setAttribute("role", "status");
    const saveRow = element("div", "rm-save-row");
    saveRow.append(save, settingsState);
    form.append(switchRow, autoStatus, gpu.row, full.row, saveRow);
    automatic.append(form);

    const history = section("最近操作");
    const operations = element("div", "rm-operations");
    const empty = element("p", "rm-empty", "暂无释放记录");
    history.append(empty, operations);
    const info = element("details", "rm-info");
    info.append(element("summary", "", "关于资源释放"), element("p", "rm-hint",
        "释放请求会等待任务队列空闲。GPU 缓存池包含 GPU 张量占用；这里不包含其他程序的显存。第三方节点持有的内存可能继续保留。"), element("p", "rm-hint",
        "操作记录显示请求状态与内存观测。ComfyUI 不提供完成回执，请结合占用变化判断结果。"));
    root.append(header, connectionError, actionMessage, memory, manual, automatic, history, info);

    function updateControls() {
        const unavailable = working || latest === null;
        const waiting = latest?.operations.some((op) => op.state === "waiting_for_idle");
        unload.disabled = release.disabled = unavailable || waiting;
        save.disabled = unavailable || !dirty;
        switchToggle.disabled = unavailable || !latest?.managed_switch_supported;
        for (const control of [gpu, full]) {
            control.toggle.disabled = unavailable;
            control.input.disabled = unavailable || !control.toggle.checked;
        }
        for (const record of records.values()) record.cancel.disabled = unavailable;
    }

    function setMessage(text, error = false) {
        actionMessage.textContent = text;
        actionMessage.hidden = !text;
        actionMessage.classList.toggle("rm-error", error);
    }

    function readSettings(settings) {
        switchToggle.checked = settings.unload_on_model_switch ?? true;
        for (const [control, seconds] of [[gpu, settings.unload_gpu_seconds], [full, settings.release_models_seconds]]) {
            control.toggle.checked = seconds !== null;
            control.input.value = String(seconds ?? control.defaultSeconds);
        }
    }

    function renderStatus(status) {
        latest = status;
        state.textContent = status.monitor_error ? "监控异常" : status.busy ? "任务进行中" : "空闲";
        state.dataset.state = status.monitor_error ? "error" : status.busy ? "busy" : "idle";
        connectionError.hidden = !status.monitor_error;
        connectionError.textContent = status.monitor_error ? `自动释放监控已停止：${status.monitor_error}` : "";
        const sample = status.memory;
        values[0].textContent = displayBytes(sample.cuda_allocated_bytes);
        values[1].textContent = displayBytes(sample.cuda_reserved_bytes);
        values[2].textContent = displayBytes(sample.rss_bytes);
        queue.textContent = status.busy
            ? `${status.running_tasks} 个任务运行中 · ${status.pending_queue} 个排队 · 释放将在空闲时进行`
            : sample.cuda_initialized ? "队列空闲" : "队列空闲 · CUDA 尚未初始化";
        if (!dirty) readSettings(status.settings);
        switchHint.textContent = status.managed_switch_supported
            ? "下次运行前释放旧模型，保留当前工作流仍需的模型。支持原生 Checkpoint、UNet、CLIP、VAE 加载器。"
            : "当前核心缺少模型切换接口，请应用插件附带的 ComfyUI 补丁后重启。";
        const timers = [status.settings.unload_gpu_seconds, status.settings.release_models_seconds];
        if (timers.every((seconds) => seconds === null)) {
            autoStatus.textContent = "空闲释放未启用 · 开启下方选项并保存后生效";
        } else if (status.monitor_error) {
            autoStatus.textContent = "监控异常 · 请重启 ComfyUI 后重试";
        } else if (status.busy) {
            autoStatus.textContent = "已启用 · 任务结束后开始计时";
        } else {
            const elapsed = status.idle_since_ms === null ? 0 : Math.max(0, (sample.sampled_at_ms - status.idle_since_ms) / 1000);
            const upcoming = timers.filter((seconds) => seconds !== null && seconds > elapsed);
            autoStatus.textContent = upcoming.length
                ? `已启用 · 下个空闲阈值还有 ${Math.ceil(Math.min(...upcoming) - elapsed)} 秒`
                : "已启用 · 已达到空闲阈值";
        }
        const labels = {
            waiting_for_idle: "等待空闲",
            dispatched: "已发送请求",
            flags_consumed_unconfirmed: "已取走请求 · 待确认效果",
            cancelled: "已取消",
        };
        const ids = new Set(status.operations.map((op) => op.id));
        for (const [id, record] of records) {
            if (!ids.has(id)) { record.row.remove(); records.delete(id); }
        }
        empty.hidden = status.operations.length > 0;
        for (const op of status.operations) {
            let record = records.get(op.id);
            if (!record) {
                const row = element("article", "rm-operation");
                const title = element("div", "rm-operation-title");
                const name = op.action === "unload_gpu" ? "卸载 GPU 模型" : "释放模型与缓存";
                title.append(element("strong", "", name), element("span", "rm-muted", op.source === "idle" ? "自动" : "手动"));
                const label = element("p", "rm-hint");
                const details = element("details", "rm-observation");
                const summary = element("summary", "", `查看占用变化 · #${op.id}`);
                const data = element("p", "rm-sample");
                details.append(summary, data);
                const cancel = button(`rm-cancel-${op.id}`, "取消等待");
                cancel.classList.add("rm-button-small");
                cancel.onclick = () => run(cancel, () => request(`release/${op.id}`, "DELETE"), "已取消等待");
                row.append(title, label, cancel, details);
                operations.prepend(row);
                record = { row, label, cancel, data };
                records.set(op.id, record);
            }
            record.label.textContent = labels[op.state];
            record.cancel.hidden = op.state !== "waiting_for_idle";
            const after = op.after;
            record.data.textContent = [
                `GPU 张量：${displayBytes(op.before.cuda_allocated_bytes)} → ${displayBytes(after?.cuda_allocated_bytes ?? null)}`,
                `GPU 缓存池：${displayBytes(op.before.cuda_reserved_bytes)} → ${displayBytes(after?.cuda_reserved_bytes ?? null)}`,
                `进程内存：${displayBytes(op.before.rss_bytes)} → ${displayBytes(after?.rss_bytes ?? null)}`,
                after ? "发送前 → 后续观测；非完成回执" : "尚无后续观测",
            ].join("\n");
        }
        updateControls();
    }

    async function refresh() {
        const id = ++refreshId;
        const status = await request("status");
        if (!stopped && id === refreshId) renderStatus(status);
    }

    async function run(control, action, message) {
        if (working || stopped) return;
        working = true;
        refreshId++;
        updateControls();
        control.setAttribute("aria-busy", "true");
        setMessage("");
        try {
            const result = await action();
            if (stopped) return;
            setMessage(typeof message === "function" ? message(result) : message);
            try { await refresh(); } catch (error) { showConnectionError(error); }
        } catch (error) {
            if (!stopped) setMessage(error.message, true);
        } finally {
            working = false;
            control.removeAttribute("aria-busy");
            if (!stopped) updateControls();
        }
    }

    function showConnectionError(error) {
        if (stopped) return;
        latest = null;
        state.textContent = "连接中断";
        state.dataset.state = "error";
        connectionError.hidden = false;
        connectionError.textContent = `无法读取资源状态：${error.message}。正在自动重试。`;
        updateControls();
    }

    form.addEventListener("input", () => {
        dirty = true;
        settingsState.textContent = "未保存";
        setMessage("");
        updateControls();
    });
    form.onsubmit = (event) => {
        event.preventDefault();
        if (!form.reportValidity()) return;
        run(save, async () => {
            const settings = await request("settings", "POST", {
                unload_gpu_seconds: gpu.toggle.checked ? Number(gpu.input.value) : null,
                release_models_seconds: full.toggle.checked ? Number(full.input.value) : null,
                unload_on_model_switch: switchToggle.checked,
            });
            dirty = false;
            readSettings(settings);
            settingsState.textContent = "已保存";
        }, "自动释放设置已保存");
    };
    const releaseMessage = (op) => op.state === "waiting_for_idle"
        ? "请求已排队，将在任务结束后释放。" : "释放请求已发送，可在最近操作中查看占用变化。";
    unload.onclick = () => run(unload, () => request("release", "POST", { action: "unload_gpu" }), releaseMessage);
    release.onclick = () => run(release, () => request("release", "POST", { action: "release_models" }), releaseMessage);
    updateControls();
    async function tick() {
        if (!working) {
            try { await refresh(); } catch (error) { showConnectionError(error); }
        }
        if (!stopped) poll = setTimeout(tick, 1500);
    }
    tick();
    return () => {
        stopped = true;
        clearTimeout(poll);
        root.remove();
        container.classList.remove("rm-host");
    };
}

app.registerExtension({
    name: "ResourceManager.Controls",
    setup() {
        const style = document.createElement("link");
        style.rel = "stylesheet";
        style.href = new URL("./resource-manager.css", import.meta.url).href;
        document.head.append(style);
        let dispose;
        app.extensionManager.registerSidebarTab({
            id: "resource-manager",
            icon: "pi pi-server",
            title: "资源管理",
            tooltip: "资源管理",
            type: "custom",
            render(container) {
                dispose?.();
                dispose = mount(container);
            },
            destroy() {
                dispose?.();
                dispose = null;
            },
        });
    },
});
