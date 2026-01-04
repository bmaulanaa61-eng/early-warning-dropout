from pydantic import BaseModel

class StudentData(BaseModel):
    Status_Pernikahan: int
    Kewarganegaraan: int
    Jenis_Kelamin: int
    Usia_Saat_Daftar: int
    Jalur_Pendaftaran: int
    Urutan_Pilihan: int
    Program_Studi: int
    Kelas_Siang_Malam: int
    Pendidikan_Sebelumnya: int
    Nilai_Pendidikan_Sebelumnya: float
    Nilai_Masuk: float
    Pendidikan_Ibu: int
    Pendidikan_Ayah: int
    Pekerjaan_Ibu: int
    Pekerjaan_Ayah: int
    Pindahan: int
    Berkebutuhan_Khusus: int
    Punya_Tunggakan: int
    SPP_Lunas: int
    Penerima_Beasiswa: int
    Mahasiswa_Internasional: int
    SKS_Sem1_Diakui: int
    SKS_Sem1_Diambil: int
    SKS_Sem1_Evaluasi: int
    SKS_Sem1_Lulus: int
    Nilai_Sem1: float
    SKS_Sem1_Tanpa_Evaluasi: int


class PredictionResponse(BaseModel):
    prediction: str
    probability_dropout: float
    probability_non_dropout: float
