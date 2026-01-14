from plant_health_app.models import DiseaseLibrary
from django.contrib.auth.models import User

# Lấy user admin (hoặc để null nếu không cần created_by)
try:
    admin_user = User.objects.get(username='admin')  # Thay 'admin' bằng username thực tế
except User.DoesNotExist:
    admin_user = None
    print("Không tìm thấy user 'admin'. Gán created_by = null.")

# Dữ liệu mẫu dựa trên PLANT_TYPE_MAPPING
diseases_data = [
    {
        "name": "apple scab",
        "plant_type": "Apple",
        "description": "Bệnh ghẻ táo gây ra các đốm đen hoặc nâu trên lá và quả táo, làm giảm chất lượng quả.",
        "symptoms": "Đốm đen hoặc nâu trên lá, quả bị biến dạng, vỏ quả nứt nẻ.",
        "treatment": "Phun thuốc trừ nấm chứa myclobutanil hoặc captan vào mùa xuân. Loại bỏ lá rụng để giảm nguồn lây nhiễm. Đảm bảo cây được thông thoáng.",
        "created_by": admin_user,
        "image": None
    },
    {
        "name": "cedar apple rust",
        "plant_type": "Apple",
        "description": "Bệnh rỉ sắt táo do nấm gây ra, thường xuất hiện trên lá và quả táo.",
        "symptoms": "Đốm vàng cam trên lá, có thể có các chấm đen nhỏ ở mặt dưới lá.",
        "treatment": "Sử dụng thuốc trừ nấm chứa sulfur hoặc myclobutanil. Loại bỏ cây tuyết tùng gần đó vì là vật chủ phụ.",
        "created_by": admin_user,
        "image": None
    },
    {
        "name": "black rot",
        "plant_type": "Grape",
        "description": "Bệnh thối đen trên nho, gây đốm đen trên lá và quả, làm quả khô héo.",
        "symptoms": "Đốm đen nhỏ trên lá, lan rộng thành vết cháy. Quả nho thối và khô lại như nho khô.",
        "treatment": "Phun thuốc chứa captan hoặc myclobutanil. Loại bỏ quả và lá bị nhiễm. Giữ vườn thông thoáng để giảm độ ẩm.",
        "created_by": admin_user,
        "image": None
    },
    {
        "name": "bacterial spot",
        "plant_type": "Pepper",
        "description": "Bệnh đốm vi khuẩn trên ớt, gây ra bởi vi khuẩn Xanthomonas.",
        "symptoms": "Đốm nhỏ màu nâu đen trên lá, quả có vết loét. Lá có thể vàng và rụng.",
        "treatment": "Sử dụng thuốc chứa đồng (copper-based). Loại bỏ cây bị nhiễm nặng. Tránh tưới nước từ trên cao.",
        "created_by": admin_user,
        "image": None
    },
    {
        "name": "cercospora leaf spot gray leaf spot",
        "plant_type": "Corn",
        "description": "Bệnh đốm lá Cercospora trên ngô, gây ra các đốm xám hoặc nâu trên lá.",
        "symptoms": "Đốm xám hoặc nâu trên lá, thường hình chữ nhật, xuất hiện ở lá thấp trước.",
        "treatment": "Sử dụng giống ngô kháng bệnh. Phun thuốc trừ nấm chứa azoxystrobin. Luân canh cây trồng.",
        "created_by": admin_user,
        "image": None
    },
    {
        "name": "common rust",
        "plant_type": "Corn",
        "description": "Bệnh rỉ sắt thông thường trên ngô, do nấm Puccinia sorghi gây ra.",
        "symptoms": "Đốm cam hoặc nâu trên lá, phân bố rải rác, có thể làm giảm quang hợp.",
        "treatment": "Phun thuốc trừ nấm chứa triazole hoặc strobilurin. Sử dụng giống kháng bệnh.",
        "created_by": admin_user,
        "image": None
    },
    {
        "name": "early blight",
        "plant_type": "Potato",
        "description": "Bệnh sớm trên khoai tây, gây đốm lá màu nâu đen với các vòng đồng tâm.",
        "symptoms": "Đốm lá màu nâu đen, thường có vòng đồng tâm giống 'mắt bò'. Lá có thể vàng và rụng sớm.",
        "treatment": "Sử dụng thuốc trừ nấm chứa chlorothalonil hoặc mancozeb. Loại bỏ lá bị nhiễm và đảm bảo luân canh cây trồng. Tưới nước vào buổi sáng.",
        "created_by": admin_user,
        "image": None
    },
    {
        "name": "esca (black measles)",
        "plant_type": "Grape",
        "description": "Bệnh esca trên nho, gây thiệt hại nghiêm trọng cho cây lâu năm.",
        "symptoms": "Lá có đốm nâu hoặc đỏ, quả bị đốm đen. Cây có thể chết dần.",
        "treatment": "Cắt tỉa cành bị nhiễm. Không có thuốc chữa hoàn toàn, nhưng phun thuốc chứa fosetyl-Al có thể giảm triệu chứng.",
        "created_by": admin_user,
        "image": None
    },

    {
        "name": "healthy",
        "plant_type": "Unknown",
        "description": "Cây khỏe mạnh, không có dấu hiệu bệnh.",
        "symptoms": "Lá xanh, không có đốm, không vàng úa, cây phát triển bình thường.",
        "treatment": "Tiếp tục chăm sóc cây bằng cách tưới nước đều đặn, bón phân cân đối và kiểm tra định kỳ để phát hiện sớm bệnh.",
        "created_by": admin_user,
        "image": None
    },
    {
        "name": "late blight",
        "plant_type": "Tomato",
        "description": "Bệnh mốc sương trên cà chua, do nấm Phytophthora infestans gây ra.",
        "symptoms": "Đốm nâu đen trên lá, thân, và quả. Lá có thể có lớp mốc trắng ở mặt dưới.",
        "treatment": "Phun thuốc chứa mancozeb hoặc metalaxyl. Loại bỏ cây bị nhiễm nặng. Tránh tưới nước từ trên cao.",
        "created_by": admin_user,
        "image": None
    },
    {
        "name": "leaf blight (isariopsis leaf spot)",
        "plant_type": "Grape",
        "description": "Bệnh đốm lá trên nho, gây ra bởi nấm Isariopsis.",
        "symptoms": "Đốm nâu hoặc xám trên lá, có thể lan rộng làm lá rụng.",
        "treatment": "Phun thuốc trừ nấm chứa captan. Loại bỏ lá bị nhiễm và giữ vườn thông thoáng.",
        "created_by": admin_user,
        "image": None
    },
    {
        "name": "leaf mold",
        "plant_type": "Tomato",
        "description": "Bệnh mốc lá trên cà chua, thường xuất hiện trong điều kiện ẩm ướt.",
        "symptoms": "Đốm vàng trên mặt trên lá, mặt dưới có lớp mốc xám hoặc trắng.",
        "treatment": "Sử dụng thuốc trừ nấm chứa chlorothalonil. Tăng thông gió trong nhà kính và giảm độ ẩm.",
        "created_by": admin_user,
        "image": None
    },
    {
        "name": "leaf scorch",
        "plant_type": "Unknown",
        "description": "Bệnh cháy lá, thường do thiếu nước hoặc vi khuẩn.",
        "symptoms": "Lá có mép cháy hoặc nâu, có thể lan rộng ra toàn lá.",
        "treatment": "Tưới nước đầy đủ và đều đặn. Nếu do vi khuẩn, sử dụng thuốc chứa đồng. Loại bỏ lá bị nhiễm.",
        "created_by": admin_user,
        "image": None
    },
    {
        "name": "northern leaf blight",
        "plant_type": "Corn",
        "description": "Bệnh đốm lá miền Bắc trên ngô, do nấm Exserohilum turcicum gây ra.",
        "symptoms": "Đốm dài, màu xám hoặc nâu trên lá, thường xuất hiện ở lá thấp.",
        "treatment": "Sử dụng giống ngô kháng bệnh. Phun thuốc chứa azoxystrobin hoặc propiconazole.",
        "created_by": admin_user,
        "image": None
    },
    {
        "name": "powdery mildew",
        "plant_type": "Cherry",
        "description": "Bệnh phấn trắng trên anh đào, do nấm Podosphaera gây ra.",
        "symptoms": "Lớp bột trắng trên lá, chồi và quả. Lá có thể xoăn lại.",
        "treatment": "Phun thuốc chứa sulfur hoặc myclobutanil. Loại bỏ lá bị nhiễm và giữ cây thông thoáng.",
        "created_by": admin_user,
        "image": None
    },
    {
        "name": "septoria leaf spot",
        "plant_type": "Tomato",
        "description": "Bệnh đốm lá Septoria trên cà chua, do nấm Septoria lycopersici gây ra.",
        "symptoms": "Đốm nâu với tâm xám trên lá, thường ở lá thấp. Lá có thể vàng và rụng.",
        "treatment": "Phun thuốc chứa chlorothalonil hoặc mancozeb. Loại bỏ lá bị nhiễm và luân canh cây trồng.",
        "created_by": admin_user,
        "image": None
    },
    {
        "name": "spider mites two-spotted spider mite",
        "plant_type": "Tomato",
        "description": "Sâu hại do nhện chích hút, gây thiệt hại trên cà chua.",
        "symptoms": "Đốm vàng nhỏ trên lá, có mạng nhện mịn. Lá có thể khô và rụng.",
        "treatment": "Phun thuốc trừ sâu chứa abamectin hoặc dầu neem. Tăng độ ẩm để hạn chế nhện phát triển.",
        "created_by": admin_user,
        "image": None
    },
    {
        "name": "target spot",
        "plant_type": "Tomato",
        "description": "Bệnh đốm mục tiêu trên cà chua, do nấm Corynespora cassiicola gây ra.",
        "symptoms": "Đốm nâu đen với vòng đồng tâm trên lá, quả bị tổn thương.",
        "treatment": "Phun thuốc chứa azoxystrobin hoặc chlorothalonil. Loại bỏ lá bị nhiễm và giữ cây thông thoáng.",
        "created_by": admin_user,
        "image": None
    },
    {
        "name": "tomato mosaic virus",
        "plant_type": "Tomato",
        "description": "Bệnh virus khảm cà chua, lây lan qua hạt và côn trùng.",
        "symptoms": "Lá xoăn, có đốm xanh vàng. Cây còi cọc, quả ít.",
        "treatment": "Loại bỏ cây bị nhiễm. Sử dụng giống kháng bệnh và khử trùng dụng cụ làm vườn.",
        "created_by": admin_user,
        "image": None
    },
    {
        "name": "tomato yellow leaf curl virus",
        "plant_type": "Tomato",
        "description": "Bệnh virus xoăn vàng lá cà chua, lây lan qua bọ phấn trắng.",
        "symptoms": "Lá xoăn, vàng, cây còi cọc, quả nhỏ và ít.",
        "treatment": "Kiểm soát bọ phấn trắng bằng thuốc chứa imidacloprid. Loại bỏ cây bị nhiễm và sử dụng giống kháng bệnh.",
        "created_by": admin_user,
        "image": None
    },
]