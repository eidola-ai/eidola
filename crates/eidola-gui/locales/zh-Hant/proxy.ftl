# 設定 ▸ 代理 —— 本機推論代理（src/proxy_settings.rs），繁體中文。

proxy-lead = 讓這台電腦上的其他工具使用你透過 Eidola 取用的模型。它們說 OpenAI API；每個請求仍然經過 Eidola —— 經過證明、由你的錢包付費，並寫入記錄。

proxy-serve = 處理請求
proxy-serve-name = 透過 Eidola 處理本機請求

proxy-listening = 正在監聽 { $address }
proxy-stopped = 未監聽
proxy-listen-failed = 無法監聽該位址 —— { $reason }

proxy-exposed-warning = 該位址可從你的網路存取，而代理尚未加密。經由它傳輸的一切 —— 你的提示詞與回答 —— 都是明文。

proxy-address = 位址
proxy-address-name = 代理監聽的 IP 位址
proxy-port-name = 代理監聽的連接埠
proxy-binding-change = 變更…
proxy-binding-save = 儲存
proxy-binding-cancel = 取消

proxy-backends = 後端
proxy-backends-note = 只有你在這裡勾選的才可存取。工具指定其他任何內容都會被告知該模型不存在。
proxy-backend-name = 透過代理提供 { $backend }
proxy-backends-empty = 尚未設定任何後端。

proxy-exposure = 裝置端模型
proxy-exposure-loaded = 僅已載入
proxy-exposure-downloaded = 全部已下載
proxy-exposure-note = 「全部已下載」會在第一個指定該模型的請求到來時啟動引擎，這需要一段時間並占用記憶體。「僅已載入」只提供正在執行的模型。

proxy-keys = API 金鑰
proxy-keys-note = 工具以 bearer 權杖傳送金鑰。Eidola 只保存它的雜湊，因此金鑰只顯示一次，之後無法再次顯示。
proxy-keys-empty = 尚無金鑰 —— 在你建立之前，任何東西都無法存取代理。
proxy-key-label-placeholder = 什麼將使用這個金鑰？
proxy-key-create = 產生金鑰
proxy-key-revoked = 已撤銷
proxy-key-unused = 從未使用
proxy-key-used = 已使用
proxy-key-revoke = 撤銷
proxy-key-revoke-name = 撤銷名為 { $label } 的金鑰

proxy-key-minted = 現在就複製。Eidola 只保存了它的雜湊，無法再次顯示。
proxy-key-copy = 複製
proxy-key-done = 完成

proxy-failed = 無法讀取代理的設定。
proxy-retry = 重試
