# 設定 ▸ 代理 —— 本機推論代理（src/proxy_settings.rs），繁體中文。

proxy-lead = 讓這台電腦上的其他工具使用你透過 Eidola 取用的模型。它們說 OpenAI API，它們請求的每次補全都會寫入記錄。其餘取決於請求前往何處：Eidola 託管服務經過證明，由你的錢包付費；裝置端模型在這台電腦上執行，不產生費用；你自己新增的伺服器會被直接存取，沒有證明，適用該伺服器自己的條款。

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
proxy-backends-note = 只有你在這裡勾選的才可存取。工具指定其他任何內容都會被告知該模型不存在。請求模型清單不會寫入記錄；對於你自己新增的伺服器，它會用該伺服器的金鑰向其索取模型目錄。
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
proxy-key-creating = 正在產生…
proxy-key-show-first = 請先複製上面的金鑰並按「完成」。
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
proxy-keys-failed = 無法列出 API 金鑰。
proxy-backends-failed = 無法讀取後端登錄。
proxy-loading = 載入中…
proxy-stale = 無法重新整理 — 顯示的是上次的結果。

proxy-error-not-an-address = { $value } 不是 IP 位址。代理監聽的是位址，而不是名稱。
proxy-error-no-port = 代理需要一個連接埠。連接埠 0 會隨意佔用一個空閒的連接埠，那不是能告訴工具的位址。
proxy-error-key-needs-name = 金鑰需要一個名稱，以便你日後分辨是哪個工具持有它。
proxy-error-cannot-listen = 無法在 { $address } 上監聽 —— { $reason }
proxy-error-stopped-accepting = 代理已停止接受連線 —— { $reason }
proxy-error-about = { $subject }：{ $message }
proxy-error-dismiss-name = 關閉關於「{ $subject }」的訊息
