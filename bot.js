const { Client, GatewayIntentBits } = require('discord.js');
require('dotenv').config();  // Untuk membaca file .env
const express = require('express');

// Inisialisasi client dengan intents
const client = new Client({
    intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages,
        GatewayIntentBits.MessageContent,
        GatewayIntentBits.DirectMessages,
        GatewayIntentBits.GuildMembers
    ]
});

// Ambil token dari environment variable
const TOKEN = process.env.DISCORD_BOT_TOKEN;

// Tabel konversi angka ke huruf (0-9 → a-j)
const number_to_letter = { "0": "a", "1": "b", "2": "c", "3": "d", "4": "e", "5": "f", "6": "g", "7": "h", "8": "i", "9": "j" };

// Karakter untuk noise (tanpa a-j)
const characters = "klmnopqrstuvwxyzKLMNOPQRSTUVWXYZ0123456789";

// Fungsi membuat string acak
function generate_random_string(length) {
    let result = "";
    for (let i = 0; i < length; i++) {
        const random_index = Math.floor(Math.random() * characters.length);
        result += characters[random_index];
    }
    return result;
}

// Fungsi konversi tanggal ke kode huruf
function convert_date_to_code(date_input) {
    let code = "";
    for (let i = 0; i < date_input.length; i++) {
        code += number_to_letter[date_input[i]];
    }
    return code;
}

// Fungsi untuk menyisipkan noise
function insert_date_code_with_noise(date_code) {
    let noise_start = generate_random_string(19).split("");
    let noise_middle = generate_random_string(19).split("");
    let noise_end = generate_random_string(18).split("");

    const pos1 = Math.floor(Math.random() * 7) + 1;
    const pos2 = Math.floor(Math.random() * 7) + 7;
    const pos3 = Math.floor(Math.random() * 7) + 1;
    const pos4 = Math.floor(Math.random() * 7) + 7;
    const pos5 = Math.floor(Math.random() * 15);
    const pos6 = Math.floor(Math.random() * 15);

    noise_start.splice(pos1, 0, date_code[0]);
    noise_start.splice(pos2, 0, date_code[1]);
    noise_middle.splice(pos3, 0, date_code[2]);
    noise_middle.splice(pos4, 0, date_code[3]);
    noise_end.splice(pos5, 0, date_code[4]);
    noise_end.splice(pos6, 0, date_code[5]);
    noise_end.push(date_code[6]);
    noise_end.splice(noise_end.length - 2, 0, date_code[7]);

    return noise_start.join("") + noise_middle.join("") + noise_end.join("");
}

// Event saat bot siap
client.once('ready', () => {
    console.log(`✅ Bot berhasil terhubung sebagai ${client.user.tag}`);
});

// Event untuk menangkap perintah
client.on('messageCreate', async message => {
    if (message.author.bot) return;

    if (message.content.startsWith('!generate')) {
        const args = message.content.split(' ');

        if (args.length < 3) {
            return message.reply('Format salah! Gunakan format: `!generate [tanggal] [username]`');
        }

        const date_input = args[1];
        const target_username = args[2];

        if (!date_input || date_input.length !== 8 || isNaN(date_input)) {
            return message.reply('Format tanggal salah! Gunakan format: `27042025`');
        }

        const target_user = message.guild.members.cache.find(member =>
            member.user.tag === target_username
        );

        if (!target_user) {
            return message.reply(`User dengan username \`${target_username}\` tidak ditemukan.`);
        }

        const date_code = convert_date_to_code(date_input);
        const license = insert_date_code_with_noise(date_code);

        try {
            const day = date_input.slice(0, 2);
            const month = date_input.slice(2, 4);
            const year = date_input.slice(4, 8);

            await target_user.send(
                `Valid Until ${day}/${month}/${year}\n` +
                ` Lisensi: \`${license}\``
            );
            message.reply(`Lisensi berhasil dikirim ke ${target_username} melalui DM.`);
        } catch (error) {
            message.reply(`Gagal mengirim DM ke ${target_username}. Pastikan mereka mengizinkan DM dari server ini.`);
        }
    }
});

// Login bot
client.login(TOKEN);

// ------------------------
// EXPRESS SERVER UNTUK HEALTH CHECK
// ------------------------
const app = express();
const PORT = process.env.PORT || 8000;

// Dummy route untuk Health Check
app.get('/', (req, res) => {
    res.send('Bot is running!');
});

// Jalankan server untuk Health Check
app.listen(PORT, () => {
    console.log(`Health check server running on port ${PORT}`);
});
