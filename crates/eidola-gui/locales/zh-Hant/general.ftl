# 設定 ▸ 一般 — 外觀、文字大小、啟動與命令列（src/general.rs），繁體中文。

general-appearance = 外觀

general-day-night = 晝夜
general-day-night-auto = 自動
general-day-night-system = 系統
general-day-night-day = 白天
general-day-night-night = 夜間
general-day-night-choice = 晝夜：{ $choice }
general-day-night-note = 「系統」跟隨 macOS。「自動」跟隨太陽 — 日出到日落之間為白天。

general-time-of-day = 時段
general-on = 開
general-off = 關
general-time-of-day-choice = 時段：{ $choice }
general-time-of-day-note = 跟隨陽光的變化 — 日出前後偏冷，正午和午夜中性，日落前後偏暖。日出和日落時間依你的時區估算。

general-light = 光線
general-light-cool = 冷
general-light-neutral = 中性
general-light-warm = 暖
general-light-choice = 光線：{ $choice }
general-light-note = 為配色指定固定的光線特徵。

general-text-size = 文字大小
general-text-size-smaller-name = 縮小文字，目前 { $percent }
general-text-size-larger-name = 放大文字，目前 { $percent }
general-text-size-reset = 重設
general-text-size-reset-name = 將文字大小重設為 100%
general-text-size-note = 縮放所有文字。也可在 View 選單中使用 — Actual Size、Zoom In、Zoom Out。

general-startup = 啟動
general-open-at-login = 登入時打開
general-login-item-switch = 登入時啟動 Eidola
general-login-item-name = 登入時打開 Eidola
general-login-item-off = 在你打開之前，Eidola 不會出現。
general-login-item-on = Eidola 隨你的工作階段一起打開。
general-login-item-needs-approval = 已開啟，但 macOS 正在等你確認 — 請在「系統設定 → 一般 → 登入項目」中允許 Eidola。
general-login-item-unsupported = 無法使用 — 需要 macOS 13 或以上版本，以及已安裝並簽署的 App。
general-login-item-unsupported-linux = 此處無法使用 — 在 Linux 上，`eidola service install` 會啟用使用者服務。
general-login-item-too-old = 登入時打開需要 macOS 13 或以上版本 — 這台 Mac 的系統較舊。
general-login-item-not-here = 此平台不支援登入時打開。

general-command-line = 命令列
general-cli-field = 終端機指令
general-cli-not-installed = 尚未安裝。安裝後會在 { $dir } 中放置一個連結到這份 Eidola 的 `eidola` 指令，可以在「終端機」中執行。
general-cli-needs-admin = macOS 會要求輸入管理者密碼。
general-cli-installed = 已安裝 — 在「終端機」中執行 `eidola` 就是執行這份 Eidola，經由 { $path }。
general-cli-dangling = { $path } 指向一份已不存在的 Eidola（{ $target }）— App 可能被移動過。更新連結以使用這份 Eidola。
general-cli-other-copy = { $path } 執行的是另一份 Eidola，位於 { $target }。更新連結以使用這份 Eidola。
general-cli-occupied-file = { $path } 已經是一個並非 Eidola 放置的檔案，Eidola 不會取代它。如果不再需要，請自行移動或刪除，然後再安裝。
general-cli-occupied-directory = { $path } 是一個檔案夾，Eidola 不會在此放置指令。如果不再需要，請自行移動或刪除，然後再安裝。
general-cli-occupied-link = { $path } 是指向 { $target } 的連結，它不是 Eidola，Eidola 不會取代它。如果不再需要，請自行移動或刪除，然後再安裝。
general-cli-occupied-other = { $path } 已被並非 Eidola 放置的內容佔用，Eidola 不會取代它。
general-cli-unreadable = 無法檢查 { $path } — { $reason }
general-cli-not-bundled = 只有當 Eidola 以已安裝的 App 執行時才能使用。
general-cli-translocated = macOS 正從暫存位置執行這份 Eidola，指向它的連結將會失效。請將 Eidola 移到「應用程式」檔案夾，從那裡打開，然後再安裝。
general-cli-missing = 這份 Eidola 中沒有命令列工具。
general-cli-install = 安裝
general-cli-install-admin = 安裝…
general-cli-install-name = 在 { $dir } 中安裝 eidola 指令
general-cli-update = 更新連結
general-cli-update-admin = 更新連結…
general-cli-update-name = 讓 { $path } 指向這份 Eidola
general-cli-installing = 正在安裝…
general-cli-waiting = 正在等待管理者密碼…
general-cli-declined = 沒有做任何更改。若要自行安裝，請在「終端機」中執行：
general-cli-failed = 無法安裝指令 — { $reason }
general-cli-manual = 若要自行安裝，請在「終端機」中執行：
general-cli-copy = 拷貝
general-cli-copy-name = 拷貝安裝指令
