# 设置 ▸ 代理 —— 本地推理代理（src/proxy_settings.rs），简体中文。

proxy-lead = 让这台电脑上的其他工具使用你通过 Eidola 访问的模型。它们说 OpenAI API；每个请求仍然经过 Eidola —— 经过证明、由你的钱包付费，并写入记录。

proxy-serve = 处理请求
proxy-serve-name = 通过 Eidola 处理本地请求

proxy-listening = 正在监听 { $address }
proxy-stopped = 未监听
proxy-listen-failed = 无法监听该地址 —— { $reason }

proxy-exposed-warning = 该地址可从你的网络访问，而代理尚未加密。经由它传输的一切 —— 你的提示词与回答 —— 都是明文。

proxy-address = 地址
proxy-address-name = 代理监听的 IP 地址
proxy-port-name = 代理监听的端口
proxy-binding-change = 更改…
proxy-binding-save = 保存
proxy-binding-cancel = 取消

proxy-backends = 后端
proxy-backends-note = 只有你在这里勾选的才可访问。工具指定其他任何内容都会被告知该模型不存在。
proxy-backend-name = 通过代理提供 { $backend }
proxy-backends-empty = 尚未配置任何后端。

proxy-exposure = 设备端模型
proxy-exposure-loaded = 仅已加载
proxy-exposure-downloaded = 全部已下载
proxy-exposure-note = “全部已下载”会在第一个指定该模型的请求到来时启动引擎，这需要一段时间并占用内存。“仅已加载”只提供正在运行的模型。

proxy-keys = API 密钥
proxy-keys-note = 工具以 bearer 令牌发送密钥。Eidola 只保存它的哈希，因此密钥只显示一次，之后无法再次显示。
proxy-keys-empty = 尚无密钥 —— 在你创建之前，任何东西都无法访问代理。
proxy-key-label-placeholder = 什么将使用这个密钥？
proxy-key-create = 生成密钥
proxy-key-creating = 正在生成…
proxy-key-show-first = 请先复制上面的密钥并按“完成”。
proxy-key-revoked = 已吊销
proxy-key-unused = 从未使用
proxy-key-used = 已使用
proxy-key-revoke = 吊销
proxy-key-revoke-name = 吊销名为 { $label } 的密钥

proxy-key-minted = 现在就复制。Eidola 只保存了它的哈希，无法再次显示。
proxy-key-copy = 复制
proxy-key-done = 完成

proxy-failed = 无法读取代理的设置。
proxy-retry = 重试
proxy-keys-failed = 无法列出 API 密钥。
proxy-backends-failed = 无法读取后端注册表。
proxy-loading = 加载中…
proxy-stale = 无法刷新 — 显示的是上次的结果。
